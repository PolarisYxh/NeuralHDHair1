import trimesh
hair_path="home/algo/yangxinhang/NeuralHDHair/NeuralHaircut/exps_second_stage/second_stage_person_0/person_0/neural_strands_w_camera_fitted/2024-01-23_20:31:36/meshes/00016000_strands_points.ply"
pcl = trimesh.load(hair_path, file_type="ply")
