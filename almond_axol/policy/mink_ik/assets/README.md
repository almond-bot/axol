# Axol Mink model

`axol_mink.urdf` and `meshes/` provide the self-contained model used by the
Mink solver. The numerical reference fixture records hashes of the URDF and
every asset, so geometry changes require an explicit reference update.

The model's root frame differs from the Axol world frame by a 90-degree yaw
about the root-joint origin at approximately `(0, 0, 0.86)` metres.
`pose6_world_to_model` converts Cartesian targets at that boundary. The joint
chains, end-effector frames and collision geometry stay in the model frame
throughout each solve. Replacing this asset with a world-frame URDF without
updating the conversion would rotate all targets incorrectly.

The bundled meshes include the finger geometry and fixed gripper/TCP frames.
Mesh references in the URDF use `package://assembly/meshes/<name>.stl`;
`ik_mujoco_model.load_mj_model` supplies MuJoCo compiler settings that resolve
these basenames in `assets/meshes` and preserve the fixed frames. The loader
does not modify the asset on disk.

`meshes/LICENSE-yam_linear4310.txt` contains the attribution and license for
the three i2rt YAM meshes. Keep it with those assets.
