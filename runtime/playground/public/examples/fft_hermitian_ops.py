import json

import torch

torch.manual_seed(0)

max_diff = 0.0
worst = ""


def check(name, d):
    global max_diff, worst
    if d > max_diff:
        max_diff = d
        worst = name


# ---------- hfft / ihfft (1-D) ----------
x = torch.randn(8)
h_rt = torch.fft.hfft(torch.fft.ihfft(x, n=8), n=8)
check("hfft_ihfft_roundtrip", (h_rt - x).abs().max().item())
# hfft output must be purely real
Hc = torch.complex(torch.randn(5), torch.randn(5))
H = torch.fft.hfft(Hc, n=8)
check("hfft_shape", abs(H.shape[0] - 8))
# ihfft output length is n // 2 + 1
ih = torch.fft.ihfft(x, n=8)
check("ihfft_len", abs(ih.shape[0] - (8 // 2 + 1)))

# ---------- irfftn / irfft2 ----------
x2 = torch.randn(4, 6)
rt_n = torch.fft.irfftn(torch.fft.rfftn(x2, dim=(-2, -1)), s=(4, 6), dim=(-2, -1))
check("irfftn_roundtrip", (rt_n - x2).abs().max().item())
rt_2 = torch.fft.irfft2(torch.fft.rfft2(x2), s=(4, 6))
check("irfft2_roundtrip", (rt_2 - x2).abs().max().item())

# ---------- hfftn / hfft2 / ihfftn / ihfft2 ----------
rt_hn = torch.fft.hfftn(torch.fft.ihfftn(x2, s=(4, 6), dim=(-2, -1)), s=(4, 6), dim=(-2, -1))
check("hfftn_roundtrip", (rt_hn - x2).abs().max().item())
rt_h2 = torch.fft.hfft2(torch.fft.ihfft2(x2, s=(4, 6)), s=(4, 6))
check("hfft2_roundtrip", (rt_h2 - x2).abs().max().item())

# ---------- ortho normalization is self-inverse for hfft ----------
ho = torch.fft.hfft(torch.fft.ihfft(x, n=8, norm="ortho"), n=8, norm="ortho")
check("hfft_ortho_roundtrip", (ho - x).abs().max().item())

ok = max_diff < 1e-2
assert ok, f"max_diff {max_diff} worst {worst}"
print(json.dumps({"ok": ok, "max_diff": max_diff, "worst": worst}, sort_keys=True))
