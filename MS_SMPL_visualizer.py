import mujoco
import mujoco.viewer
import open3d as o3d
import numpy as np

# load and visualize the scaled model in MuJoCo to check for any obvious issues
def visualize_mujoco_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            print(data.qpos)  # should be near zero/neutral


def visualize_mesh_and_joints(path_to_smpl, path_to_scaled_model, z_offset=0.0, x_offset=0.0):
    # Load SMPL mesh (Y-up, pelvis at origin by SMPL convention)
    o3d_mesh = o3d.io.read_triangle_mesh(path_to_smpl)
    smpl_verts = np.asarray(o3d_mesh.vertices)

    # MuJoCo joint positions in Z-up world frame
    model = mujoco.MjModel.from_xml_path(path_to_scaled_model)
    data = mujoco.MjData(model)

    def set_joint(name, value):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value

    # Raise arms to T-pose: elv_angle selects coronal plane, shoulder_elv lifts to 90°
    set_joint("elv_angle_r", -np.pi/10)
    set_joint("shoulder_elv_r", 51*np.pi / 100)
    set_joint("elv_angle_l", -np.pi/10)
    set_joint("shoulder_elv_l", 51*np.pi / 100)

    mujoco.mj_kinematics(model, data)
    joint_positions = data.xpos.copy()  # (nbody, 3), index 0 is worldbody

    # Pelvis world position — body 1 in this model
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mujoco_pelvis = joint_positions[pelvis_id]

    # SMPL: Y-up, body faces -Z (back at +Z), anatomical right at +X.
    # MuJoCo world: Z-up, body faces +X, anatomical right at -Y.
    # Combined mapping: SMPL(x,y,z) → MuJoCo(-z, -x, y)
    R = np.array([[ 0, 0, -1],
                  [-1, 0,  0],
                  [ 0, 1,  0]])
    Rotation_z90 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                            [np.sin(np.pi), np.cos(np.pi), 0],
                            [0, 0, 1]])
    smpl_verts_rot = (Rotation_z90 @ R @ smpl_verts.T).T

    # SMPL pelvis is at the origin; after rotation it stays at (0,0,0).
    # Translate so SMPL root lands on the MuJoCo pelvis world position.
    smpl_verts_aligned = smpl_verts_rot + mujoco_pelvis + np.array([x_offset, 0, z_offset])

    aligned_mesh = o3d.geometry.TriangleMesh()
    aligned_mesh.vertices = o3d.utility.Vector3dVector(smpl_verts_aligned)
    aligned_mesh.triangles = o3d_mesh.triangles
    aligned_mesh.compute_vertex_normals()
    aligned_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # Joint spheres — already in MuJoCo world space, skip worldbody (index 0)
    spheres = []
    for pos in joint_positions[1:]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(pos)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        spheres.append(sphere)

    o3d.visualization.draw_geometries([aligned_mesh] + spheres)

    

if __name__ == "__main__":
    visualize_mesh_and_joints("./smpl_shaped.obj", "./MS-Human-700.xml", z_offset=0.20, x_offset=-0.05)