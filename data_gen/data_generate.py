from srmt.planning_scene import PlanningScene, VisualSimulator
import srmt.planning_scene.planning_scene_tools as PlanningSceneTools
import numpy as np
import random
from math import sqrt, pi
import time
import matplotlib.pyplot as plt
import scipy.spatial.transform as sci_tf
import os
import datetime as dt
import sys
import pickle
import argparse
import itertools

# np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)
np.set_printoptions(threshold=sys.maxsize)
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}


def main(args):
    # Parameters
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min 
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max
    shelf_pert_pose = np.array([-0.1, -0.1, -pi/6], # min (x, y, theta)
                               [ 0.3,  0.1,  pi/6]) # max (x, y, theta)
    
    # Create Planning Scene
    pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="base")

    # Create cameras
    vs1 = VisualSimulator(n_grid=args.num_grid)
    vs2 = VisualSimulator(n_grid=args.num_grid)
    vs3 = VisualSimulator(n_grid=args.num_grid)
    vs4 = VisualSimulator(n_grid=args.num_grid)

    dataset1 = {} # for grid
    dataset2 = {} # for depth

    depth_set = []
    voxel_set = []
    normal_q_set = []
    nerf_q_set= []
    coll_set  = []
    min_dist_set = []
    env_idx_set = []

    for iter in range(args.num_env):
        
        # Create shelf
        PlanningSceneTools.add_shelf(pc=pc, 
                                pos=np.array([0 + random.triangular(shelf_pert_pose[0,0], shelf_pert_pose[1,0]),
                                              0 + random.triangular(shelf_pert_pose[0,1], shelf_pert_pose[1,1]),
                                              0.5 + 1.006]),
                                dphi=0,
                                dtheta=-pi/2 +random.triangular(shelf_pert_pose[0,2], shelf_pert_pose[1,2]),
                                length=1.0,
                                width=0.3,
                                height=1.0,
                                d=0.05,
                                shelf_parts=3,
                                id=0)

        vs1.load_scene(pc)
        vs2.load_scene(pc)
        vs3.load_scene(pc)
        vs4.load_scene(pc)

        r = 1.5
        vs1.set_cam_and_target_pose(np.array([-r/2,  r*sqrt(3)/2, 0.63 + 1.006    ]), np.array([0, 0, 0.63 + 1.006])) 
        vs2.set_cam_and_target_pose(np.array([-r/2, -r*sqrt(3)/2, 0.63 + 1.006    ]), np.array([0, 0, 0.63 + 1.006]))
        vs3.set_cam_and_target_pose(np.array([-r,    1e-8,        0.63 + 1.006    ]), np.array([0, 0, 0.63 + 1.006]))
        vs4.set_cam_and_target_pose(np.array([ 1e-8, 0,           r + 0.63 + 1.006]), np.array([0, 0, 0.63 + 1.006]))


        depth1 = vs1.generate_depth_image()
        depth2 = vs2.generate_depth_image()
        depth3 = vs3.generate_depth_image()
        depth4 = vs4.generate_depth_image()

        depths = np.stack([depth1, depth2, depth3, depth4])
        dataset1["depth_len"] = depths.shape[-1]
        dataset1["depth_size"] = depths.size
        depth_set.append(depths)

        scene_bound_min = np.array([-r, -r, -r + 0.63 + 1.006])
        scene_bound_max = np.array([ r,  r,  r + 0.63 + 1.006])

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


        q_res = args.q_res * (pi/180)
        for q in np.array(itertools.product( np.concatenate(np.arange(joint_limit[0,0], joint_limit[1,0], q_res), [joint_limit[1,0]]),
                                             np.concatenate(np.arange(joint_limit[0,1], joint_limit[1,1], q_res), [joint_limit[1,1]]),
                                             np.concatenate(np.arange(joint_limit[0,2], joint_limit[1,2], q_res), [joint_limit[1,2]]),
                                             np.concatenate(np.arange(joint_limit[0,3], joint_limit[1,3], q_res), [joint_limit[1,3]]),
                                             np.concatenate(np.arange(joint_limit[0,4], joint_limit[1,4], q_res), [joint_limit[1,4]]),
                                             np.concatenate(np.arange(joint_limit[0,5], joint_limit[1,5], q_res), [joint_limit[1,5]]),
                                             np.concatenate(np.arange(joint_limit[0,6], joint_limit[1,6], q_res), [joint_limit[1,6]]) )):
            min_dist = pc.min_distance(q)

            if min_dist == -1: # collide
                normal_q_set.append( np.array([(q[q_idx] - joint_limit[0, q_idx]) / (joint_limit[1, q_idx] - joint_limit[0, q_idx]) for q_idx in range(7)]) )
                nerf_q_set.append( np.concatenate([q, np.cos(q), np.sin(q)], axis=0) )
                coll_set.append(1)
                min_dist_set.append(min_dist)
                env_idx_set.append(iter)
            elif min_dist > 0.05 and min_dist < 0.08:
                normal_q_set.append( np.array([(q[q_idx] - joint_limit[0, q_idx]) / (joint_limit[1, q_idx] - joint_limit[0, q_idx]) for q_idx in range(7)]) )
                nerf_q_set.append( np.concatenate([q, np.cos(q), np.sin(q)], axis=0) )
                coll_set.append(0)
                min_dist_set.append(min_dist) 
                env_idx_set.append(iter)


        t1 = time.time()

    date = dt.datetime.now()
    data_dir = "data/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)

    if args.data_type == "both":
        with open(data_dir + "/box_depth.pickle", "wb") as f:
            dataset1["depth"] = np.array(depth_set)
            dataset1["normalize_q"] = np.array(normal_q_set)
            dataset1["nerf_q"] = np.array(nerf_q_set)
            dataset1["coll"] = np.array(coll_set)
            dataset1["min_dist"] = np.array(min_dist_set)
            dataset1["env_idx"] = np.array(env_idx_set)
            pickle.dump(dataset1,f)
        with open(data_dir + "/box_grid.pickle", "wb") as f:
            dataset2["grid"] = np.array(voxel_set)
            dataset2["normalize_q"] = np.array(normal_q_set)
            dataset2["nerf_q"] = np.array(nerf_q_set)
            dataset2["coll"] = np.array(coll_set)
            dataset2["min_dist"] = np.array(min_dist_set)
            dataset2["env_idx"] = np.array(env_idx_set)
            pickle.dump(dataset2,f)
    elif args.data_type == "depth":
        with open(data_dir + "/box_depth.pickle", "wb") as f:
            dataset1["depth"] = np.array(depth_set)
            dataset1["normalize_q"] = np.array(normal_q_set)
            dataset1["nerf_q"] = np.array(nerf_q_set)
            dataset1["coll"] = np.array(coll_set)
            dataset1["min_dist"] = np.array(min_dist_set)
            dataset1["env_idx"] = np.array(env_idx_set)
            pickle.dump(dataset1,f)
    elif args.data_type == "grid":
        with open(data_dir + "/box_grid.pickle", "wb") as f:
            dataset2["grid"] = np.array(voxel_set)
            dataset2["normalize_q"] = np.array(normal_q_set)
            dataset2["nerf_q"] = np.array(nerf_q_set)
            dataset2["coll"] = np.array(coll_set)
            dataset2["min_dist"] = np.array(min_dist_set)
            dataset2["env_idx"] = np.array(env_idx_set)
            pickle.dump(dataset2,f)
    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"num_env": args.num_env,
                  "num_grid": args.num_grid,
                  "q_res": args.q_res,
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
    parser.add_argument("--num_grid", type=int, default=32)
    parser.add_argument("--q_res", type=float, default=20)
    parser.add_argument("--data_type", type=str, default="both", help="grid / depth / both")

    args = parser.parse_args()
    main(args)

