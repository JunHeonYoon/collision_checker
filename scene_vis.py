from srmt.planning_scene import PlanningScene, VisualSimulator
import srmt.planning_scene.planning_scene_tools as PlanningSceneTools
import numpy as np
from math import pi, sqrt
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
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])
    panda_joint_init = np.array([0.0, 0.0, 0.0, -1.0471, 0.0, 1.0471, 0.7853])

    # Create Planning Scene
    pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="base")

    # Create cameras
    vs1 = VisualSimulator(n_grid=args.num_grid)
    vs2 = VisualSimulator(n_grid=args.num_grid)
    vs3 = VisualSimulator(n_grid=args.num_grid)
    vs4 = VisualSimulator(n_grid=args.num_grid)
    

    # Create shelf
    PlanningSceneTools.add_shelf(pc=pc, 
                             pos=np.array([-0.1,0.1,0.5 + 1.00]),
                             dphi=0,
                             dtheta=-pi/2+pi/6,
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
    vs1.set_cam_and_target_pose(np.array([-r/2,  r*sqrt(3)/2, 0.63 + 1    ]), np.array([0, 0, 0.63 + 1])) 
    vs2.set_cam_and_target_pose(np.array([-r/2, -r*sqrt(3)/2, 0.63 + 1    ]), np.array([0, 0, 0.63 + 1]))
    vs3.set_cam_and_target_pose(np.array([-r,    1e-8,        0.63 + 1    ]), np.array([0, 0, 0.63 + 1]))
    vs4.set_cam_and_target_pose(np.array([ 1e-8, 0,           r + 0.63 + 1]), np.array([0, 0, 0.63 + 1]))


    depth1 = vs1.generate_depth_image()
    depth2 = vs2.generate_depth_image()
    depth3 = vs3.generate_depth_image()
    depth4 = vs4.generate_depth_image()

    scene_bound_min = np.array([-r, -r, -r + 0.63 + 1])
    scene_bound_max = np.array([ r,  r,  r + 0.63 + 1])

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


    pc.display(panda_joint_init)
    time.sleep(1)

    ax1 = plt.figure(1).add_subplot(231)
    ax1.set_title("depth image1", fontsize=16, fontweight='bold', pad=20)
    ax1.imshow(depth1)
    ax2 = plt.figure(1).add_subplot(232)
    ax2.set_title("depth image2", fontsize=16, fontweight='bold', pad=20)
    ax2.imshow(depth2)
    ax3 = plt.figure(1).add_subplot(233)
    ax3.set_title("depth image3", fontsize=16, fontweight='bold', pad=20)
    ax3.imshow(depth3)
    ax4 = plt.figure(1).add_subplot(234)
    ax4.set_title("depth image4", fontsize=16, fontweight='bold', pad=20)
    ax4.imshow(depth4)


    ax9 = plt.figure(2).add_subplot(projection='3d')
    ax9.voxels(voxel_grids)
    ax9.set_title("voxel grid all", fontsize=16, fontweight='bold', pad=20)
    plt.show()

    while(True):
        q = np.random.uniform(low=joint_limit[0,:], high=joint_limit[1,:])
        # pc.display(q)
        min_dist = pc.min_distance_vector(q)
        min_dist = np.min(min_dist)
        min_dist_ = pc.min_distance(q)
        print(min_dist)
        print(min_dist_)
        print("\t\t")
        if min_dist_ == -1:
            print("collide")
        time.sleep(1)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_grid", type=int, default=64)
    args = parser.parse_args()
    main(args)

