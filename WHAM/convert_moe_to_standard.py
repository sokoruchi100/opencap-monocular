"""
Convert MoE_wholebody.pth -> wholebody.pth for use with standard (non-MoE) ViTPose.

The MoE_wholebody.pth checkpoint was trained with a MoE (Mixture of Experts)
backbone where each transformer block's MLP has 6 expert sub-layers
(backbone.blocks.*.mlp.experts.*) in addition to the standard fc1/fc2 layers.

The standard ViTPose code only expects fc1 and fc2. Since these shared weights
are already present in the MoE checkpoint alongside the experts, we can simply
strip the expert keys and save - no approximation or averaging needed.

Result: wholebody.pth with 133-keypoint output head, compatible with:
  configs/wholebody/.../ViTPose_huge_wholebody_256x192.py
"""

import torch

input_path = "MoE_wholebody.pth"
output_path = "wholebody.pth"

ckpt = torch.load(input_path, map_location="cpu")
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Drop the MoE expert keys — the standard fc1/fc2 weights are already present
new_state = {k: v for k, v in state.items() if ".mlp.experts." not in k}

print(f"Original keys : {len(state)}")
print(f"Converted keys: {len(new_state)}  ({len(state) - len(new_state)} expert keys removed)")

for k in new_state:
    if "final_layer" in k:
        print(f"{k}: {new_state[k].shape}")

torch.save(new_state, output_path)
print(f"Saved to {output_path}")
