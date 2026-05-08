import pickle
import numpy as np
import torch
import xml.etree.ElementTree as ET
import os
import shutil

def extract_SMPL_length(beta):
    with open("./WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl", "rb") as f:
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


if __name__ == "__main__":
    beta_torch = torch.zeros(10, dtype=torch.float32)
    J, height, shaped_mesh, faces = extract_SMPL_length(beta_torch)
    save_mesh_obj(shaped_mesh, faces, "MS-Scaled/smpl_shaped.obj")
    SMPL_bone_lengths = compute_segment_lengths(J)
    root, scale_factors = compute_scale_factors(SMPL_bone_lengths)
    path_to_scaled_xml = scale_MS_model(root, scale_factors)
    validate_scaled_model(SMPL_bone_lengths, path_to_scaled_xml)