import json
import torch
import torch.nn.functional as F

x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]])

nearest = F.interpolate(x, size=(6, 8), mode="nearest")
bilinear = F.interpolate(x, size=(6, 8), mode="bilinear", align_corners=False)
bilinear_ac = F.interpolate(x, size=(6, 8), mode="bilinear", align_corners=True)

out = {
    "input": x.tolist(),
    "nearest_values": nearest.tolist(),
    "bilinear_values": bilinear.tolist(),
    "bilinear_align_corners_values": bilinear_ac.tolist(),
    "nearest_shape": list(nearest.shape),
    "bilinear_shape": list(bilinear.shape),
}
print(json.dumps(out, indent=2))
