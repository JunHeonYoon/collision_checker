from srmt.planning_scene import PlanningScene, VisualSimulator
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import scipy.spatial.transform as sci_tf
import os
import datetime as dt
import sys
import pickle
import argparse

NUM_LINK = 9

# np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)
np.set_printoptions(threshold=sys.maxsize)
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}


def main(args):
    # Parameters
    workspace = np.array([0.855, 1.19]) # radius, z
    robot_basespace = np.array([0.2, 1.19]) # radius, z
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])
    obs_size_limit = np.array([[0.1, 0.1, 0.1],
                            [0.2, 0.2, 0.2]])
    panda_posi = np.array([0.0, 0.0, 1.006])

    # Create Planning Scene
    pc = PlanningScene(arm_names=["panda_arm"], arm_dofs=[7], base_link="base")
    # pc = PlanningScene(arm_names=["panda_arm_1", "panda_arm_2"], arm_dofs=[7,7], base_link="base")

    # Create cameras
    vs1 = VisualSimulator(n_grid=args.num_grid)
    vs2 = VisualSimulator(n_grid=args.num_grid)
    vs3 = VisualSimulator(n_grid=args.num_grid)
    vs4 = VisualSimulator(n_grid=args.num_grid)

    dataset1 = {}
    dataset2 = {}

    depth_set = []
    voxel_set = []
    nerf_q_set= []
    coll_set  = []
    for iter in range(args.num_env):
        
        # Create obstacles
        for i in range(args.num_ob):
            obs_posi_p = [random.triangular(robot_basespace[0], workspace[0]), 
                        random.triangular(-math.pi, math.pi), 
                        random.triangular(0, workspace[1])] # r, theta, z
            obs_posi = [obs_posi_p[0]*math.cos(obs_posi_p[1]), obs_posi_p[0]*math.sin(obs_posi_p[1]), obs_posi_p[2]] + panda_posi
            obs_size = obs_size_limit[0]+ np.random.random(3) * ( obs_size_limit[1] - obs_size_limit[0] )
            obs_ori = sci_tf.Rotation.from_euler('z', random.triangular(-math.pi, math.pi)).as_quat()
            pc.add_box('ob_{}'.format(i), obs_size.tolist(), obs_posi, obs_ori)

        vs1.load_scene(pc)
        vs2.load_scene(pc)
        vs3.load_scene(pc)
        vs4.load_scene(pc)

        vs1.set_cam_and_target_pose(np.array([2.5*workspace[0]*math.cos(math.pi*0/3), 2.5*workspace[0]*math.sin(math.pi*0/3), 0.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)
        vs2.set_cam_and_target_pose(np.array([2.5*workspace[0]*math.cos(math.pi*2/3), 2.5*workspace[0]*math.sin(math.pi*2/3), 0.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)
        vs3.set_cam_and_target_pose(np.array([2.5*workspace[0]*math.cos(math.pi*4/3), 2.5*workspace[0]*math.sin(math.pi*4/3), 0.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)
        vs4.set_cam_and_target_pose(np.array([0.0001,                                 0,                                      2.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)


        depth1 = vs1.generate_depth_image()
        depth2 = vs2.generate_depth_image()
        depth3 = vs3.generate_depth_image()
        depth4 = vs4.generate_depth_image()

        depths = np.stack([depth1, depth2, depth3, depth4])
        dataset1["depth_len"] = depths.shape[-1]
        dataset1["depth_size"] = depths.size
        depth_set.append(depths)

        scene_bound_min = np.array([-workspace[0], -workspace[0], 0]) + panda_posi
        scene_bound_max = np.array([ workspace[0],  workspace[0], workspace[1]]) + panda_posi

        vs1.set_scene_bounds(scene_bound_min, scene_bound_max)
        vs2.set_scene_bounds(scene_bound_min, scene_bound_max)
        vs3.set_scene_bounds(scene_bound_min, scene_bound_max)
        vs4.set_scene_bounds(scene_bound_min, scene_bound_max)

        voxel_grid1 = vs1.generate_voxel_occupancy()
        voxel_grid2 = vs2.generate_voxel_occupancy()
        voxel_grid3 = vs3.generate_voxel_occupancy()
        voxel_grid4 = vs4.generate_voxel_occupancy()

        voxel_grid1 = voxel_grid1.reshape(args.num_grid, args.num_grid, args.num_grid)
        voxel_grid2 = voxel_grid2.reshape(args.num_grid, args.num_grid, args.num_grid)
        voxel_grid3 = voxel_grid3.reshape(args.num_grid, args.num_grid, args.num_grid)
        voxel_grid4 = voxel_grid4.reshape(args.num_grid, args.num_grid, args.num_grid)

        voxel_grids = np.any(np.array([voxel_grid1, voxel_grid2, voxel_grid3, voxel_grid4]), axis=0).astype(int)
        dataset2["grid_len"] = voxel_grids.shape[0]
        dataset2["grid_size"] = voxel_grids.size
        voxel_set.append(voxel_grids)

        coll_type_num = [[0, 0, 0, 0] for _ in range(NUM_LINK)] 
        t0 = time.time()
        while np.sum(np.array(coll_type_num)) < np.sum([ int(args.num_q / NUM_LINK * args.coll_ratio[_]) for _ in range(4) ]) * NUM_LINK and time.time() - t0 < 600:
            joint_state = joint_limit[0] + np.random.random(7) *( joint_limit[1] - joint_limit[0] )
            # joint_state_1 = joint_limit[0] + np.random.random(7) *( joint_limit[1] - joint_limit[0] )
            # joint_state_2 = joint_limit[0] + np.random.random(7) *( joint_limit[1] - joint_limit[0] )
            # joint_state = np.concatenate([joint_state_1, joint_state_2], axis=0)
            pc.display(joint_state)
            min_dist = pc.min_distance_vector(joint_state) 
            coll = np.where(min_dist > 0, 0, 1) # 1 is collide, 0 is non-collide

            for link_idx in range(NUM_LINK):
                if min_dist[link_idx] == -1: # collide
                    if coll_type_num[link_idx][0] < int(args.num_q / NUM_LINK * args.coll_ratio[0]):
                        nerf_q_set.append(np.concatenate([joint_state, np.cos(joint_state), np.sin(joint_state)],axis=0))
                        coll_set.append(coll)
                        coll_type_num[link_idx][0] += 1
                        continue
                elif min_dist[link_idx] <= 0.01:  # surface
                    if coll_type_num[link_idx][1] < int(args.num_q / NUM_LINK * args.coll_ratio[1]):
                        nerf_q_set.append(np.concatenate([joint_state, np.cos(joint_state), np.sin(joint_state)],axis=0))
                        coll_set.append(coll)
                        coll_type_num[link_idx][1] += 1
                        continue
                elif min_dist[link_idx] <= 0.1:  # close
                    if coll_type_num[link_idx][2] < int(args.num_q / NUM_LINK * args.coll_ratio[2]):
                        nerf_q_set.append(np.concatenate([joint_state, np.cos(joint_state), np.sin(joint_state)],axis=0))
                        coll_set.append(coll)
                        coll_type_num[link_idx][2] += 1
                        continue
                else:  # far
                    if coll_type_num[link_idx][3] < int(args.num_q / NUM_LINK * args.coll_ratio[3]):
                        nerf_q_set.append(np.concatenate([joint_state, np.cos(joint_state), np.sin(joint_state)],axis=0))
                        coll_set.append(coll)
                        coll_type_num[link_idx][3] += 1
                        continue

        t1 = time.time()
        print("Data generate: {0:0.1f} % completed! ({1:.1f} sec)".format((iter+1)/args.num_env*100, t1 - t0))
        print("Data percentage: {0:0.1f} %".format( np.sum(np.array(coll_type_num)) / (np.sum([ int(args.num_q / NUM_LINK * args.coll_ratio[_]) for _ in range(4) ]) * NUM_LINK) * 100 ) )

    date = dt.datetime.now()
    data_dir = "data/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)

    if args.data_type == "depth":
        with open(data_dir + "/box_depth.pickle", "wb") as f:
            dataset1["depth"] = np.array(depth_set)
            dataset1["nerf_q"] = np.array(nerf_q_set)
            dataset1["coll"] = np.array(coll_set)
            pickle.dump(dataset1,f)
    elif args.data_type == "grid":
        with open(data_dir + "/box_grid.pickle", "wb") as f:
            dataset2["grid"] = np.array(voxel_set)
            dataset2["nerf_q"] = np.array(nerf_q_set)
            dataset2["coll"] = np.array(coll_set)
            pickle.dump(dataset2,f)
    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"num_env": args.num_env,
                  "num_ob": args.num_ob,
                  "num_grid": args.num_grid,
                  "num_q": args.num_q,
                  "coll_ratio": args.coll_ratio,
                  "data_type": args.data_type}
        for param, value in params.items():
            f.write(f'{param} : {value}\n')

    import shutil
    folder_path = "data/"
    num_save = 3
    order_list = sorted(os.listdir(folder_path), reverse=True)[1:]
    remove_folder_list = order_list[num_save:]
    for rm_folder in remove_folder_list:
        shutil.rmtree(folder_path+rm_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_env", type=int, default=100)
    parser.add_argument("--num_ob", type=int, default=10)
    parser.add_argument("--num_grid", type=int, default=32)
    parser.add_argument("--num_q", type=int, default=10000)
    parser.add_argument("--coll_ratio", type=float, nargs=4, default=[0.4, 0.2, 0.2, 0.2], help="0: collide / 1: surface (0.01m) / 2: close (0.1m) / 3: far")
    parser.add_argument("--data_type", type=str, default="grid", help="grid / depth")

    args = parser.parse_args()
    main(args)

