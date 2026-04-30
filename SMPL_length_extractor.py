import pickle
import numpy as np
import torch

def extract_SMPL_length(beta):
    with open("./WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl", "rb") as f:
        SMPL = pickle.load(f, encoding="latin1")

    v_template = np.array(SMPL.get('v_template'))
    shapedirs = np.array(SMPL.get('shapedirs'))
    J_regressor = np.array(SMPL.get('J_regressor').todense()) # convert from sparse to dense

    # Convert beta to numpy array and pad it
    beta_np = beta.detach().cpu().numpy()
    beta_padded = np.zeros(300, dtype=np.float32)
    beta_padded[:10] = beta_np

    shaped_mesh = v_template + shapedirs @ beta_padded
    J = J_regressor @ shaped_mesh
    height = shaped_mesh[411, 1] - shaped_mesh[:, 1].min()
    return J, height

def bone_length(joints, i, j):
    return np.linalg.norm(joints[i] - joints[j])

def compute_segment_lengths(joints):
    bone_lengths = {}

    # Femur (average left and right)
    bone_lengths["femur"] = (bone_length(joints, 2, 5) + bone_length(joints, 1, 4)) / 2

    # Tibia
    bone_lengths["tibia"] = (bone_length(joints, 5, 8) + bone_length(joints, 4, 7)) / 2

    # Foot
    bone_lengths["foot"] = (bone_length(joints, 8, 11) + bone_length(joints, 7, 10)) / 2

    # Clavicle
    bone_lengths["clavicle"] = (bone_length(joints, 13, 16) + bone_length(joints, 14, 17)) / 2

    # Humerus
    bone_lengths["humerus"] = (bone_length(joints, 17, 19) + bone_length(joints, 16, 18)) / 2

    # Radius
    bone_lengths["radius"] = (bone_length(joints, 19, 21) + bone_length(joints, 18, 20)) / 2

    # Hand
    bone_lengths["hand"] = (bone_length(joints, 21, 23) + bone_length(joints, 20, 22)) / 2

    # Neck
    bone_lengths["neck"] = bone_length(joints, 12, 15)

    # Spine (sum the chain)
    bone_lengths["spine"] = (bone_length(joints, 0, 3) + 
            bone_length(joints, 3, 6) + 
            bone_length(joints, 6, 9) +
            bone_length(joints, 9, 12))

    return bone_lengths

def print_height_ratios(bone_lengths, height):
    print(f"height = {height:.2f} meters")
    for k, v in bone_lengths.items():
        print(f"{k} length = {v:.2f} meters")
        print(f"{k} ratio = {v/height*100:.2f}%")

if __name__ == "__main__":
    beta_torch = torch.zeros(10, dtype=torch.float32)
    J, height = extract_SMPL_length(beta_torch)
    bone_lengths = compute_segment_lengths(J)
    print_height_ratios(bone_lengths, height)