"""Check that the research feature extractor loads identically in Vitis AI 3.5."""

import hashlib
from pathlib import Path

import timm
import torch

weights = Path("/weights/wrn50_2_l12_timm_1.0.28.pth")
model = timm.create_model("wide_resnet50_2", pretrained=False, features_only=True, out_indices=(1, 2))
state = torch.load(weights, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()
digest = hashlib.sha256()
for name, tensor in model.state_dict().items():
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(tensor.contiguous().numpy().tobytes())
with torch.no_grad():
    outputs = model(torch.zeros(1, 3, 224, 224))
print(f"TIMM={timm.__version__} TORCH={torch.__version__}")
print(f"WEIGHT_TENSOR_SHA256={digest.hexdigest()}")
print(f"FEATURE_SHAPES={[tuple(x.shape) for x in outputs]}")
assert digest.hexdigest() == "dd5281cfe6a0f1d8fc340b15419c67486758ff2ee09cf99e90fc31dbfd60be2c"
assert [tuple(x.shape) for x in outputs] == [(1, 256, 56, 56), (1, 512, 28, 28)]
print("WRN_MODEL_LOAD_PASS")
