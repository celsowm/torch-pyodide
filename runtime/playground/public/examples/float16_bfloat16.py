import json
import torch

# float16 and bfloat16 dtype support
torch.manual_seed(42)

# Create tensors in different dtypes
a_f32 = torch.tensor([[1.5, 2.7], [3.1, 4.9]], dtype=torch.float32)
a_f16 = torch.tensor([[1.5, 2.7], [3.1, 4.9]], dtype=torch.float16)
a_bf16 = torch.tensor([[1.5, 2.7], [3.1, 4.9]], dtype=torch.bfloat16)

# Check dtype tracking
print(f"float32 dtype: {a_f32.dtype}")
print(f"float16 dtype: {a_f16.dtype}")
print(f"bfloat16 dtype: {a_bf16.dtype}")

# Operations preserve dtype
b_f16 = a_f16.add(a_f16)
print(f"float16 after add: {b_f16.dtype}")

# half() method
c_f16 = a_f32.half()
print(f"after .half(): {c_f16.dtype}")

# bfloat16() method
d_bf16 = a_f32.bfloat16()
print(f"after .bfloat16(): {d_bf16.dtype}")

# Values are close but may differ slightly due to precision
out = {
    "float32_values": a_f32.tolist(),
    "float16_values": a_f16.tolist(),
    "bfloat16_values": a_bf16.tolist(),
    "f16_dtype": str(a_f16.dtype),
    "bf16_dtype": str(a_bf16.dtype),
    "status": "OK"
}
print(json.dumps(out, indent=2))
