from srmt.planning_scene import PlanningScene, VisualSimulator
import srmt.planning_scene.planning_scene_tools as PlanningSceneTools
import numpy as np
from math import sqrt, pi
import time
import os
import datetime as dt
import sys
import pickle
import argparse
import itertools
from multiprocessing import Process, Queue

# np.printoptions(precision=3, suppress=True, linewidth=100, threshold=10000)
np.set_printoptions(threshold=sys.maxsize)
title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}


def main(args):
    # Number of threads
    if args.num_th > os.cpu_count():
        args.num_th = os.cpu_count()
        print("core: {}".format(args.num_th))

    # Parameters
    joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],  # min 
                            [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]]) # max
    np.random.seed(args.seed)
    shelf_pert_pose_set = np.random.uniform(low= [-0.1, -0.1, 0.5 + 1, -pi/6], # (x, y, z, theta)
                                            high=[ 0.3,  0.1, 0.5 + 1,  pi/6],
                                            size=[args.num_env, 4])
    
    q_res = args.q_res * (pi/180)
    q_set = list(itertools.product( np.concatenate((np.arange(joint_limit[0,0], joint_limit[1,0], q_res), [joint_limit[1,0]])),
                                    np.concatenate((np.arange(joint_limit[0,1], joint_limit[1,1], q_res), [joint_limit[1,1]])),
                                    np.concatenate((np.arange(joint_limit[0,2], joint_limit[1,2], q_res), [joint_limit[1,2]])),
                                    np.concatenate((np.arange(joint_limit[0,3], joint_limit[1,3], q_res), [joint_limit[1,3]])),
                                    np.concatenate((np.arange(joint_limit[0,4], joint_limit[1,4], q_res), [joint_limit[1,4]])),
                                    np.concatenate((np.arange(joint_limit[0,5], joint_limit[1,5], q_res), [joint_limit[1,5]])),
                                    np.concatenate((np.arange(joint_limit[0,6], joint_limit[1,6], q_res*3), [joint_limit[1,6]]))))
    q_set = np.array(q_set)
    print(q_set.shape)
    

    def work(id, start_idx, end_idx, env_idx, result):

        dataset = {}
        depth_set = []
        voxel_set = []
        normal_q_set = []
        nerf_q_set= []
        coll_set  = []
        min_dist_set = []
        env_idx_set = []

        # Create Planning Scene
        pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="base", topic_name="planning_scene_suhan" + str(id))

        # Create shelf
        PlanningSceneTools.add_shelf(pc=pc, 
                                pos=shelf_pert_pose_set[env_idx, :3],
                                # pos=[0,0,1.5],
                                dphi=0,
                                dtheta=-pi/2 +shelf_pert_pose_set[env_idx, 3],
                                # dtheta=-pi/2,
                                length=1.0,
                                width=0.3,
                                height=1.0,
                                d=0.05,
                                shelf_parts=3,
                                id=0)

        # Create cameras
        vs1 = VisualSimulator(n_grid=args.num_grid)
        vs2 = VisualSimulator(n_grid=args.num_grid)
        vs3 = VisualSimulator(n_grid=args.num_grid)
        vs4 = VisualSimulator(n_grid=args.num_grid)
        
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

        depths = np.stack([depth1, depth2, depth3, depth4])

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
        if id == 0:
            t0 = time.time()

        for iter, q in enumerate(q_set[start_idx:end_idx, :]):
            min_dist = pc.min_distance(q)
            if min_dist == -1: # collide
                normal_q_set.append( np.array([(q[q_idx] - joint_limit[0, q_idx]) / (joint_limit[1, q_idx] - joint_limit[0, q_idx]) for q_idx in range(7)]) )
                nerf_q_set.append( np.concatenate([q, np.cos(q), np.sin(q)], axis=0) )
                coll_set.append(1)
                min_dist_set.append(min_dist)
                env_idx_set.append(env_idx)
            elif min_dist > 0.05 and min_dist < 0.08: # boundary
                normal_q_set.append( np.array([(q[q_idx] - joint_limit[0, q_idx]) / (joint_limit[1, q_idx] - joint_limit[0, q_idx]) for q_idx in range(7)]) )
                nerf_q_set.append( np.concatenate([q, np.cos(q), np.sin(q)], axis=0) )
                coll_set.append(0)
                min_dist_set.append(min_dist) 
                env_idx_set.append(env_idx)   

            if id == 0:
                if (time.time() - t0) % 10 <= 0.0005:
                    print("{0:.1f} sec {1:.1f}% completed on one thread.".format(time.time()-t0, (iter+(end_idx-start_idx)*env_idx)/((end_idx-start_idx)*args.num_env)*100))

            

        if id == 0:
            dataset["grid_len"] = voxel_grids.shape[0]
            dataset["grid_size"] = voxel_grids.size
            voxel_set.append(voxel_grids)   
            dataset["depth_len"] = depths.shape[-1]
            dataset["depth_size"] = depths.size
            depth_set.append(depths)


        dataset["depth"] = depth_set
        dataset["grid"] = voxel_set
        dataset["normalize_q"] = normal_q_set
        dataset["nerf_q"] = nerf_q_set
        dataset["coll"] = coll_set
        dataset["min_dist"] = min_dist_set
        dataset["env_idx"] = env_idx_set
        dataset["id"] = id

        result.put(dataset)
        return
    
    date = dt.datetime.now()
    data_dir = "data/{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    os.mkdir(data_dir)
    t0 = time.time()

    for env_idx in range(args.num_env):
        result = Queue()
        threads =[]
        num_q = q_set.shape[0] // args.num_th
        
        dataset = {}
        dataset["depth"] = []
        dataset["grid"] = []
        dataset["normalize_q"] = []
        dataset["nerf_q"] = []
        dataset["coll"] = []
        dataset["min_dist"] = []
        dataset["env_idx"] = []
        
        for i in range(args.num_th):
            if i == args.num_th-1:
                th = Process(target=work, args=(i, i*num_q, q_set.shape[0], env_idx, result))
            else:
                th = Process(target=work, args=(i, i*num_q, (i+1)*num_q, env_idx, result))
            threads.append(th)
        
        print("Start multi-threading!")
        for i in range(args.num_th):
            threads[i].start()
        

        for i in range(args.num_th):
            data = result.get()
            id = data["id"]
            if id == 0:
                dataset["grid_len"] = data["grid_len"]
                dataset["grid_size"] = data["grid_size"]
                dataset["depth_len"] = data["depth_len"]
                dataset["depth_size"] = data["depth_size"]
                dataset["depth"] = data["depth"]
                dataset["grid"] = data["grid"]
            dataset["normalize_q"] = dataset["normalize_q"] + data["normalize_q"]
            dataset["nerf_q"] = dataset["nerf_q"] + data["nerf_q"]
            dataset["coll"] = dataset["coll"] +  data["coll"]
            dataset["min_dist"] = dataset["min_dist"] + data["min_dist"]
            dataset["env_idx"] = dataset["env_idx"] + data["env_idx"] 

        for i in range(args.num_th):
            threads[i].join()

        with open(data_dir + "/dataset" + str(env_idx) +".pickle", "wb") as f:
            dataset["depth"] = np.array(dataset["depth"])
            dataset["grid"] = np.array(dataset["grid"])
            dataset["normalize_q"] = np.array(dataset["normalize_q"])
            dataset["nerf_q"] = np.array(dataset["nerf_q"])
            dataset["coll"] = np.array(dataset["coll"])
            dataset["min_dist"] = np.array(dataset["min_dist"])
            dataset["env_idx"] = np.array(dataset["env_idx"])
            pickle.dump(dataset,f)

    print("Time: {:.2f}".format(time.time() - t0))
    
    

    with open(data_dir + "/param_setting.txt", "w", encoding='UTF-8') as f:
        params = {"num_env": args.num_env,
                  "num_grid": args.num_grid,
                  "q_res": args.q_res,
                  "data_type": args.data_type,
                  "seed": args.seed}
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
    parser.add_argument("--num_th", type=int, default=30)
    parser.add_argument("--num_env", type=int, default=10)
    parser.add_argument("--num_grid", type=int, default=32)
    parser.add_argument("--q_res", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_type", type=str, default="both", help="grid / depth / both")

    args = parser.parse_args()
    main(args)

