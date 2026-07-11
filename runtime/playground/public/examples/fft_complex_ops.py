import json
import math
import torch

torch.manual_seed(0)

# ---------- complex tensor basics ----------
re = torch.tensor([1.0, 2.0, 3.0, 4.0])
im = torch.tensor([0.5, -1.0, 0.0, 2.0])
z = torch.complex(re, im)
z_is_complex = z.is_complex()
real_ok = (z.real - re).abs().max().item() < 1e-6
imag_ok = (z.imag - im).abs().max().item() < 1e-6

# round-trip via view_as_real / view_as_complex
rr = torch.view_as_real(z)
z2 = torch.view_as_complex(rr)
view_ok = (z2.real - re).abs().max().item() < 1e-6

# complex arithmetic: (a*b) checked against manual component math
a = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, -1.0]))
b = torch.complex(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 1.0]))
prod = a * b
# (1+3i)(0+2i) = -6 + 2i ; (2-1i)(1+1i) = 3 + 1i
mul_ok = (
    abs(prod.real[0].item() + 6.0) < 1e-5
    and abs(prod.imag[0].item() - 2.0) < 1e-5
    and abs(prod.real[1].item() - 3.0) < 1e-5
    and abs(prod.imag[1].item() - 1.0) < 1e-5
)
conj_ok = (a.conj().imag + a.imag).abs().max().item() < 1e-6
abs_ok = abs(torch.complex(torch.tensor([3.0]), torch.tensor([4.0])).abs().item() - 5.0) < 1e-5

# ---------- FFT round-trips ----------
x = torch.randn(8)
X = torch.fft.fft(x)
x_rec = torch.fft.ifft(X).real
fft_roundtrip = (x_rec - x).abs().max().item()

# DFT[0] equals the sum of the signal
dc_err = abs(X.real[0].item() - x.sum().item())

# rfft / irfft round-trip
Xr = torch.fft.rfft(x)
rfft_len_ok = list(Xr.shape) == [8 // 2 + 1]
x_from_rfft = torch.fft.irfft(Xr, n=8)
irfft_roundtrip = (x_from_rfft - x).abs().max().item()

# ortho normalization keeps energy (Parseval)
Xo = torch.fft.fft(x, norm="ortho")
energy_err = abs((Xo.abs() * Xo.abs()).sum().item() - (x * x).sum().item())

# 2D fft round-trip
x2 = torch.randn(4, 6)
x2_rec = torch.fft.ifft2(torch.fft.fft2(x2)).real
fft2_roundtrip = (x2_rec - x2).abs().max().item()

# frequency helper
freqs = torch.fft.fftfreq(4)
fftfreq_ok = (freqs - torch.tensor([0.0, 0.25, -0.5, -0.25])).abs().max().item() < 1e-6

out = {
    "z_is_complex": z_is_complex,
    "real_ok": real_ok,
    "imag_ok": imag_ok,
    "view_roundtrip_ok": view_ok,
    "complex_mul_ok": mul_ok,
    "conj_ok": conj_ok,
    "abs_ok": abs_ok,
    "fft_roundtrip_err": fft_roundtrip,
    "fft_dc_err": dc_err,
    "rfft_len_ok": rfft_len_ok,
    "irfft_roundtrip_err": irfft_roundtrip,
    "ortho_energy_err": energy_err,
    "fft2_roundtrip_err": fft2_roundtrip,
    "fftfreq_ok": fftfreq_ok,
    "status": "OK",
}
print(json.dumps(out, indent=2))
