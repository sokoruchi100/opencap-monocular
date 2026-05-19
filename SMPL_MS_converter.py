import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import pickle
import numpy as np
import torch
import xml.etree.ElementTree as ET
import os
import shutil
import mujoco
import time

def extract_SMPL_length(beta, path="./WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl"):
    with open(path, "rb") as f:
        SMPL = pickle.load(f, encoding="latin1")

    v_template = np.array(SMPL.get('v_template'))
    shapedirs = np.array(SMPL.get('shapedirs'))
    J_regressor = np.array(SMPL.get('J_regressor').todense()) # convert from sparse to dense
    faces = np.array(SMPL.get('f'))

    # Convert beta to numpy array and pad it; average across frames if shape is (T, 10)
    beta_np = beta.detach().cpu().numpy() if isinstance(beta, torch.Tensor) else np.asarray(beta)
    if beta_np.ndim == 2:
        beta_np = beta_np.mean(axis=0)
    beta_padded = np.zeros(300, dtype=np.float32)
    beta_padded[:10] = beta_np

    shaped_mesh = v_template + shapedirs @ beta_padded
    J = J_regressor @ shaped_mesh
    height = shaped_mesh[411, 1] - shaped_mesh[:, 1].min()
    return J, height, shaped_mesh, faces

def save_mesh_obj(vertices, faces, path):
    """Write shaped SMPL mesh to an OBJ file (vertices in metres, Y-up)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ face indices are 1-based
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def calculate_SMPL_bone_length(joints, i, j):
    return np.linalg.norm(joints[i] - joints[j])

def compute_segment_lengths(joints):
    SMPL_bone_lengths = {}

    # Femur (average left and right)
    SMPL_bone_lengths["femur"] = (calculate_SMPL_bone_length(joints, 2, 5) + calculate_SMPL_bone_length(joints, 1, 4)) / 2

    # Tibia
    SMPL_bone_lengths["tibia"] = (calculate_SMPL_bone_length(joints, 5, 8) + calculate_SMPL_bone_length(joints, 4, 7)) / 2

    # Foot
    SMPL_bone_lengths["foot"] = (calculate_SMPL_bone_length(joints, 8, 11) + calculate_SMPL_bone_length(joints, 7, 10)) / 2

    # Clavicle
    SMPL_bone_lengths["clavicle"] = (calculate_SMPL_bone_length(joints, 13, 16) + calculate_SMPL_bone_length(joints, 14, 17)) / 2

    # Humerus
    SMPL_bone_lengths["humerus"] = (calculate_SMPL_bone_length(joints, 17, 19) + calculate_SMPL_bone_length(joints, 16, 18)) / 2

    # Radius
    SMPL_bone_lengths["radius"] = (calculate_SMPL_bone_length(joints, 19, 21) + calculate_SMPL_bone_length(joints, 18, 20)) / 2

    # # Hand
    # SMPL_bone_lengths["hand"] = (calculate_SMPL_bone_length(joints, 21, 23) + calculate_SMPL_bone_length(joints, 20, 22)) / 2

    # # Neck
    # SMPL_bone_lengths["neck"] = calculate_SMPL_bone_length(joints, 12, 15)

    # Spine (sum the chain)
    SMPL_bone_lengths["spine"] = (calculate_SMPL_bone_length(joints, 0, 3) + 
            calculate_SMPL_bone_length(joints, 3, 6) + 
            calculate_SMPL_bone_length(joints, 6, 9) +
            calculate_SMPL_bone_length(joints, 9, 12))

    return SMPL_bone_lengths

def print_height_ratios(SMPL_bone_lengths, height):
    print(f"height = {height:.2f} meters")
    for k, v in SMPL_bone_lengths.items():
        print(f"{k} length = {v:.2f} meters")
        print(f"{k} ratio = {v/height*100:.2f}%")

def load_xml_with_includes(path):
    """Recursively resolve <include> tags and return merged root."""
    tree = ET.parse(path)
    root = tree.getroot()
    base_dir = "MS-Human-700"

    for include in root.findall(".//include"):
        inc_path = os.path.join(base_dir, include.get("file"))
        inc_root = load_xml_with_includes(inc_path)
        parent = next(r for r in root.iter() if include in list(r))
        idx = list(parent).index(include)
        parent.remove(include)
        for child in list(inc_root):
            parent.insert(idx, child)
            idx += 1

    return root

def get_body_pos(body_elem):
    pos = body_elem.get("pos", "0 0 0")
    return np.array([float(x) for x in pos.split()])

def calculate_MS_bone_length(root, child_body_name):
    """Length of bone ending at child_body_name = magnitude of its pos."""
    body = root.find(f".//body[@name='{child_body_name}']")
    if body is None:
        return None
    return float(np.linalg.norm(get_body_pos(body)))

def compute_scale_factors(SMPL_bone_lengths, path="MS-Human-700/MS-Human-700.xml"):
    # load and parse the XML file containing average human body segment lengths
    root = load_xml_with_includes(path)
    
    # Define generic bone lengths
    MS_bones_parent_to_child = {
        "femur_r":    "tibia_r",
        "femur_l":    "tibia_l",
        "tibia_r":    "talus_r",
        "tibia_l":    "talus_l",
        "foot_r":     "toes_r",
        "foot_l":     "toes_l",
        "clavicle_r": "clavphant_r",
        "clavicle_l": "clavphant_l",
        "humerus_r":  "ulna_r",
        "humerus_l":  "ulna_l",
        "radius_r":   "proximal_row_r",
        "radius_l":   "proximal_row_l",
    }

    MS_spine = {
        "sacrum":     "lumbar5",
        "lumbar5":    "lumbar4",
        "lumbar4":    "lumbar3",
        "lumbar3":    "lumbar2",
        "lumbar2":    "lumbar1",
        "lumbar1":    "thoracic12",
        "thoracic12": "thoracic11",
        "thoracic11": "thoracic10",
        "thoracic10": "thoracic9",
        "thoracic9":  "thoracic8",
        "thoracic8":  "thoracic7",
        "thoracic7":  "thoracic6",
        "thoracic6":  "thoracic5",
        "thoracic5":  "thoracic4",
        "thoracic4":  "thoracic3",
        "thoracic3":  "thoracic2",
        "thoracic2":  "thoracic1",
    }

    bones_list = list(MS_bones_parent_to_child.items())
    MS_bone_lengths = {}
    for i in range(len(bones_list))[::2]:
        b_r, c_r = bones_list[i]
        b_l, c_l = bones_list[i+1]
        base_name = b_r.split("_")[0]
        MS_bone_lengths[base_name] = (calculate_MS_bone_length(root, c_r) + calculate_MS_bone_length(root, c_l)) / 2

    for b_r, c_r in MS_spine.items():
        MS_bone_lengths["spine"] = MS_bone_lengths.get("spine", 0) + calculate_MS_bone_length(root, c_r)

    # Compute scale factors to convert MS bone lengths to SMPL bone lengths
    scale_factors = {
        bone: SMPL_bone_lengths[bone] / MS_bone_lengths[bone] 
        for bone in MS_bone_lengths
    }

    return root, scale_factors

def scale_MS_model(root, scale_factors):
    # Now map the MS bones to the SMPL bone names
    MS_to_SMPL_mapping = {
        "tibia_l": "femur",
        "tibia_r": "femur",
        "talus_l": "tibia",
        "talus_r": "tibia",
        "toes_l": "foot",
        "toes_r": "foot",
        "clavphant_l": "clavicle",
        "clavphant_r": "clavicle",
        "ulna_l": "humerus",
        "ulna_r": "humerus",
        "proximal_row_l": "radius",
        "proximal_row_r": "radius",
        "lumbar5":       "spine",
        "lumbar4":       "spine",
        "lumbar3":       "spine",
        "lumbar2":       "spine",
        "lumbar1":       "spine",
        "thoracic12":    "spine",
        "thoracic11":    "spine",
        "thoracic10":    "spine",
        "thoracic9":     "spine",
        "thoracic8":     "spine",
        "thoracic7":     "spine",
        "thoracic6":     "spine",
        "thoracic5":     "spine",
        "thoracic4":     "spine",
        "thoracic3":     "spine",
        "thoracic2":     "spine",
        "thoracic1":     "spine",
    }

    # iterate over every body and scale the pos attribute by the appropriate factor
    for body in root.findall(".//body"):
        name = body.get("name")
        if name in MS_to_SMPL_mapping:
            smpl_bone = MS_to_SMPL_mapping[name]
            scale_factor = scale_factors[smpl_bone]

            # scale body position
            pos = get_body_pos(body)
            scaled_pos = pos * scale_factor
            body.set("pos", " ".join(map(str, scaled_pos)))

            # scale all site positions within this body
            for site in body.findall("site"):
                site_pos = get_body_pos(site)
                scaled_site_pos = site_pos * scale_factor
                site.set("pos", " ".join(map(str, scaled_site_pos)))

            # also scale geom positions with class wrap if they exist
            for geom in body.findall("geom[@class='wrap']"):
                geom_pos = get_body_pos(geom)
                scaled_geom_pos = geom_pos * scale_factor
                geom.set("pos", " ".join(map(str, scaled_geom_pos)))
    
    out_dir = "MS-Scaled"
    os.makedirs(out_dir, exist_ok=True)

    # Write the merged, scaled tree as a single flat XML (no includes)
    ET.ElementTree(root).write(
        os.path.join(out_dir, "MS-Human-700.xml"),
        xml_declaration=True,
        encoding="unicode",
    )

    # Copy asset files (meshes) referenced by bone geoms
    src_assets = "MS-Human-700/Asset"
    dst_assets = os.path.join(out_dir, "Asset")
    if os.path.isdir(src_assets):
        shutil.copytree(src_assets, dst_assets, dirs_exist_ok=True)

    # also copy geom assets if they exist
    src_geom_assets = "MS-Human-700/Geometry"
    dst_geom_assets = os.path.join(out_dir, "Geometry")
    if os.path.isdir(src_geom_assets):
        shutil.copytree(src_geom_assets, dst_geom_assets, dirs_exist_ok=True)
    
    # return path to the new scaled XML for verification
    return os.path.join(out_dir, "MS-Human-700.xml")

def validate_scaled_model(SMPL_bone_lengths, path):
    # Recompute the bone lengths from the scaled model and print ratios to verify correctness, should be close to 1.0 if scaling was done correctly
    print(compute_scale_factors(SMPL_bone_lengths, path)[1])

# Coordinate transform: SMPL (X=lateral, Y=up, Z=posterior) → MS ISB (X=anterior, Y=up, Z=lateral)
_T = np.array([[0., 0., 1.],
               [0., 1.,  0.],
               [1., 0.,  0.]])

# MS joint axes from XML, keyed by DOF name.
# Bilateral joints: {'l': array, 'r': array}. Spine: plain array.
MS_JOINT_AXES = {
    'hip_flexion':         {'l': np.array([ 0.,  0.,  1.]), 'r': np.array([ 0.,  0.,  1.])},
    'hip_adduction':       {'l': np.array([-1.,  0.,  0.]), 'r': np.array([ 1.,  0.,  0.])},
    'hip_rotation':        {'l': np.array([ 0., -1.,  0.]), 'r': np.array([ 0.,  1.,  0.])},
    'knee_angle':          {'l': np.array([ 0., 0.0707131,  -0.997497]), 'r': np.array([ 0.,  -0.0707131,  -0.997497])},
    'ankle_angle':         {'l': np.array([ 0.105014,  0.174022,  0.979126]), 'r': np.array([-0.105014, -0.174022,  0.979126])},
    'subtalar_angle':      {'l': np.array([-0.78718,  -0.604747, -0.120949]), 'r': np.array([ 0.78718,   0.604747, -0.120949])},
    'mtp_angle':           {'l': np.array([-0.580954,  0.,       -0.813936]), 'r': np.array([ 0.580954,  0.,       -0.813936])},
    'sternoclavicular_r2': {'l': np.array([-0.0153,   -0.989299, -0.1451  ]), 'r': np.array([ 0.0153,    0.989299, -0.1451  ])},
    'sternoclavicular_r3': {'l': np.array([ 0.994473,  0.,       -0.104997]), 'r': np.array([-0.994473,  0.,       -0.104997])},
    'elv_angle':           {'l': np.array([-0.0048,   -0.999089,  0.0424  ]), 'r': np.array([ 0.0048,    0.999089,  0.0424  ])},
    'shoulder_elv':        {'l': np.array([ 0.998261, -0.0023,    0.058898]), 'r': np.array([-0.998261,  0.0023,    0.058898])},
    'shoulder_rot':        {'l': np.array([-0.0048,   -0.999089,  0.0424  ]), 'r': np.array([ 0.0048,    0.999089,  0.0424  ])},
    'elbow_flexion':       {'l': np.array([-0.0494,   -0.0366,    0.998108]), 'r': np.array([ 0.0494,    0.0366,    0.998108])},
    'pro_sup':             {'l': np.array([ 0.017161, -0.992666, -0.119668]), 'r': np.array([-0.017161,  0.992666, -0.119668])},
    'deviation':           {'l': np.array([ 0.819064,  0.135611, -0.557444]), 'r': np.array([-0.819064, -0.135611, -0.557444])},
    'wrist_flexion':       {'l': np.array([-0.956427,  0.252207,  0.147104]), 'r': np.array([ 0.956427, -0.252207,  0.147104])},
    'spine_FE':            np.array([0., 0., 1.]),
    'spine_LB':            np.array([1., 0., 0.]),
    'spine_AR':            np.array([0., 1., 0.]),
}
# Pre-normalise all axes
for _k, _v in MS_JOINT_AXES.items():
    if isinstance(_v, dict):
        MS_JOINT_AXES[_k] = {s: a / np.linalg.norm(a) for s, a in _v.items()}
    else:
        MS_JOINT_AXES[_k] = _v / np.linalg.norm(_v)

# Rest-pose corrections: (sign, offset) applied as  angle = sign*raw + offset
# Needed where SMPL's T-pose rest differs from MS's qpos=0 rest.
# shoulder_elv: SMPL T-pose has arms horizontal (π/2 in MS); MS qpos=0 has arms hanging.
#   corrected = π/2 - raw  →  sign=-1, offset=π/2
SMPL_DOF_CORRECTIONS = {
    'shoulder_elv':  (-1, np.pi / 2),   # SMPL T-pose arms horizontal; MS rest arms-down
    'elv_angle':     ( 1, -np.pi / 10), # SMPL T-pose plane offset: coronal plane is -pi/10 in MS
    'elbow_flexion': (-1, 0.0),          # SMPL flexion sign is opposite to MS axis direction
}

# SMPL joint index → MS joint mapping.
# Each entry: (smpl_start, side, [(dof_name, ms_joint_name), ...])
# side is 'l', 'r', or None for spine.
SMPL_JOINT_MAP = [
    ( 3, 'l', [('hip_flexion', 'hip_flexion_l'), ('hip_adduction', 'hip_adduction_l'), ('hip_rotation', 'hip_rotation_l')]),
    ( 6, 'r', [('hip_flexion', 'hip_flexion_r'), ('hip_adduction', 'hip_adduction_r'), ('hip_rotation', 'hip_rotation_r')]),
    ( 9, None,[('spine_FE', 'L5_S1_FE'),   ('spine_LB', 'L5_S1_LB'),   ('spine_AR', 'L5_S1_AR')]),
    (12, 'l', [('knee_angle', 'knee_angle_l')]),
    (15, 'r', [('knee_angle', 'knee_angle_r')]),
    (18, None,[('spine_FE', 'T12_L1_FE'),  ('spine_LB', 'T12_L1_LB'),  ('spine_AR', 'T12_L1_AR')]),
    (21, 'l', [('ankle_angle', 'ankle_angle_l'), ('subtalar_angle', 'subtalar_angle_l')]),
    (24, 'r', [('ankle_angle', 'ankle_angle_r'), ('subtalar_angle', 'subtalar_angle_r')]),
    (27, None,[('spine_FE', 'T1_head_neck_FE'), ('spine_LB', 'T1_head_neck_LB'), ('spine_AR', 'T1_head_neck_AR')]),
    (30, 'l', [('mtp_angle', 'mtp_angle_l')]),
    (33, 'r', [('mtp_angle', 'mtp_angle_r')]),
    (39, 'l', [('sternoclavicular_r2', 'sternoclavicular_r2_l'), ('sternoclavicular_r3', 'sternoclavicular_r3_l')]),
    (42, 'r', [('sternoclavicular_r2', 'sternoclavicular_r2_r'), ('sternoclavicular_r3', 'sternoclavicular_r3_r')]),
    (48, 'l', [('elv_angle', 'elv_angle_l'), ('shoulder_elv', 'shoulder_elv_l'), ('shoulder_rot', 'shoulder_rot_l')]),
    (51, 'r', [('elv_angle', 'elv_angle_r'), ('shoulder_elv', 'shoulder_elv_r'), ('shoulder_rot', 'shoulder_rot_r')]),
    (54, 'l', [('elbow_flexion', 'elbow_flexion_l'), ('pro_sup', 'pro_sup_l')]),
    (57, 'r', [('elbow_flexion', 'elbow_flexion_r'), ('pro_sup', 'pro_sup_r')]),
    (60, 'l', [('deviation', 'deviation_l'), ('wrist_flexion', 'flexion_l')]),
    (63, 'r', [('deviation', 'deviation_r'), ('wrist_flexion', 'flexion_r')]),
]

def smpl_root_to_pelvis_joints(R_world, trans_world):
    """
    R_world     : (N, 3, 3) pelvis rotation matrix in SMPL world frame (poses_root_world)
    trans_world : (N, 3)    pelvis translation in SMPL world frame (trans_world)
    returns     : dict mapping MS pelvis joint names → (N,) displacement/angle arrays

    MS-Human-700 root DOF breakdown (all scalar joints):
      pelvis_tx/ty/tz  — slide joints along X/Y/Z axes
      pelvis_tilt      — hinge around Z  (frontal tilt)
      pelvis_list      — hinge around X  (lateral list)
      pelvis_rotation  — hinge around Y  (axial rotation)
    """
    R_world     = np.asarray(R_world)
    trans_world = np.asarray(trans_world, dtype=float)

    # Remap SMPL world axes to MS world axes (swaps X ↔ Z)
    trans_ms = trans_world @ _T.T   # (N, 3)

    # Rotate orientation into MS world frame, then project onto each hinge axis
    R_ms = _T @ R_world @ _T.T     # (N, 3, 3)
    rv   = Rotation.from_matrix(R_ms).as_rotvec()  # (N, 3)

    return {
        'pelvis_tx':       trans_ms[:, 0],
        'pelvis_ty':       trans_ms[:, 1],
        'pelvis_tz':       trans_ms[:, 2],
        'pelvis_tilt':     np.einsum('ni,i->n', rv, np.array([0., 0., 1.])),
        'pelvis_list':     np.einsum('ni,i->n', rv, np.array([1., 0., 0.])),
        'pelvis_rotation': np.einsum('ni,i->n', rv, np.array([0., 1., 0.])),
    }


def smpl_to_ms_joint(aa_smpl, dof_map, side=None):
    """
    aa_smpl : (N, 3) or (3,) SMPL axis-angle for one joint
    dof_map : list of (dof_name, ms_joint_name) from SMPL_JOINT_MAP
    side    : 'l', 'r', or None for spine
    returns : dict {ms_joint_name: (N,) angle array}
    """
    R    = Rotation.from_rotvec(aa_smpl).as_matrix()
    R_ms = _T @ R @ _T.T
    rv   = Rotation.from_matrix(R_ms).as_rotvec()   # (..., 3)

    result = {}
    for dof_name, ms_name in dof_map:
        ax = MS_JOINT_AXES[dof_name]
        n  = ax[side] if isinstance(ax, dict) else ax   # already normalised
        raw = np.einsum('...i,i->...', rv, n)
        sign, offset = SMPL_DOF_CORRECTIONS.get(dof_name, (1, 0.0))
        result[ms_name] = sign * raw + offset
    return result


# Build a full qpos sequence (N frames)
def build_qpos_sequence(model, joint_values):
    """
    joint_values: dict of joint_name -> (N,) array
    Unmapped DOFs are zeroed out. Free/ball joints get identity quaternions.
    returns: (N, model.nq) array
    """

    N = next(iter(joint_values.values())).shape[0]
    qpos_seq = np.zeros((N, model.nq))

    # Identity quaternion for free/ball joints (w=1, xyz=0)
    for i in range(model.njnt):
        adr = model.jnt_qposadr[i]
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_seq[:, adr + 3] = 1.0   # free joint: [tx ty tz qw qx qy qz]
        elif model.jnt_type[i] == mujoco.mjtJoint.mjJNT_BALL:
            qpos_seq[:, adr] = 1.0        # ball joint: [qw qx qy qz]

    for name, values in joint_values.items():
        try:
            adr = model.joint(name).qposadr[0]
            qpos_seq[:, adr] = values
        except KeyError:
            pass  # joint absent in this model variant

    return qpos_seq

def build_qpos_dict(smpl_pose):
    joint_values = {}
    for start, side, dof_map in SMPL_JOINT_MAP:
        joint_values.update(smpl_to_ms_joint(smpl_pose[:, start:start+3], dof_map, side=side))
    return joint_values

def get_ms_scaled_model_and_data(path_to_scaled_model):
    results = joblib.load("wham_output.pkl")
    wham_data = results[0]  # subject 0
    smpl_pose   = wham_data['pose']              # (N, 72) axis-angle
    R_world     = wham_data['poses_root_world']  # (N, 3, 3) pelvis rotation in world
    trans_world = wham_data['trans_world']       # (N, 3)   pelvis position in world

    model = mujoco.MjModel.from_xml_path(path_to_scaled_model)
    data  = mujoco.MjData(model)

    joint_values = build_qpos_dict(smpl_pose)
    joint_values.update(smpl_root_to_pelvis_joints(R_world, trans_world))
    qpos_seq = build_qpos_sequence(model, joint_values)

    return model, data, qpos_seq

if __name__ == "__main__":
    results = joblib.load("wham_output.pkl")
    data = results[0]  # subject 0
    smpl_beta = data['betas']  # (10,) shape coefficients
    smpl_pose = data['pose']  # (N, 72) axis-angle
    
    J, height, shaped_mesh, faces = extract_SMPL_length(smpl_beta, path="SMPL_NEUTRAL.pkl")
    SMPL_bone_lengths = compute_segment_lengths(J)
    root, scale_factors = compute_scale_factors(SMPL_bone_lengths)
    path_to_scaled_xml = scale_MS_model(root, scale_factors)
    validate_scaled_model(SMPL_bone_lengths, path_to_scaled_xml)