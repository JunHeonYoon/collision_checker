from srmt.planning_scene import PlanningScene, VisualSimulator
import srmt.planning_scene.planning_scene_tools as PlanningSceneTools
import numpy as np
import random
import scipy.spatial.transform as sci_tf
import math
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models_grid import CollNet
import torch
import pickle

num_ob = 10
num_grid = 32
num_link = 9

# Parameters
workspace = np.array([0.855, 1.19]) # radius, z
robot_basespace = np.array([0.4, 1.19]) # radius, z
joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                        [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])
obs_size_limit = np.array([[0.1, 0.1, 0.1],
                           [0.2, 0.2, 0.2]])
panda_posi = np.array([0.0, 0.0, 1.006])
panda_joint_init = np.array([0.0, 0.0, 0.0, -1.0471, 0.0, 1.0471, 0.7853])

# Create Planning Scene
pc = PlanningScene(arm_names=["panda_arm"], arm_dofs=[7], base_link="base")
# pc = PlanningScene(arm_names=["panda_arm_1", "panda_arm_2"], arm_dofs=[7,7], base_link="base")

# Create cameras
vs1 = VisualSimulator(n_grid=num_grid)
vs2 = VisualSimulator(n_grid=num_grid)
vs3 = VisualSimulator(n_grid=num_grid)
vs4 = VisualSimulator(n_grid=num_grid)

# Create obstacles
# for i in range(num_ob):
#     obs_posi_p = [random.triangular(robot_basespace[0], workspace[0]), 
#                 random.triangular(-math.pi, math.pi), 
#                 random.triangular(0, workspace[1])] # r, theta, z
#     obs_posi = [obs_posi_p[0]*math.cos(obs_posi_p[1]), obs_posi_p[0]*math.sin(obs_posi_p[1]), obs_posi_p[2]] + panda_posi
#     # obs_ori = sci_tf.Rotation.random().as_quat()
#     obs_size = obs_size_limit[0]+ np.random.random(3) * ( obs_size_limit[1] - obs_size_limit[0] )
#     obs_ori = sci_tf.Rotation.from_euler('z', random.triangular(-math.pi, math.pi)).as_quat()
#     pc.add_box('ob_{}'.format(i), obs_size.tolist(), obs_posi, obs_ori)

PlanningSceneTools.add_shelf(pc=pc, 
                             pos=panda_posi + np.array([0.7,0,0.5]),
                             dphi=0,
                             dtheta=-3.141592/2,
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

vs1.set_cam_and_target_pose(np.array([2.5*workspace[0]*math.cos(math.pi*0/3), 2.5*workspace[0]*math.sin(math.pi*0/3), 0.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)
vs2.set_cam_and_target_pose(np.array([2.5*workspace[0]*math.cos(math.pi*2/3), 2.5*workspace[0]*math.sin(math.pi*2/3), 0.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)
vs3.set_cam_and_target_pose(np.array([2.5*workspace[0]*math.cos(math.pi*4/3), 2.5*workspace[0]*math.sin(math.pi*4/3), 0.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)
vs4.set_cam_and_target_pose(np.array([0.0001,                                 0,                                      2.5*workspace[1]]) + panda_posi, np.array([0.0, 0.0, 0.5*workspace[1]]) + panda_posi)


depth1 = vs1.generate_depth_image()
depth2 = vs2.generate_depth_image()
depth3 = vs3.generate_depth_image()
depth4 = vs4.generate_depth_image()

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

voxel_grid1 = voxel_grid1.reshape(num_grid, num_grid, num_grid)
voxel_grid2 = voxel_grid2.reshape(num_grid, num_grid, num_grid)
voxel_grid3 = voxel_grid3.reshape(num_grid, num_grid, num_grid)
voxel_grid4 = voxel_grid4.reshape(num_grid, num_grid, num_grid)

voxel_grids = np.any(np.array([voxel_grid1, voxel_grid2, voxel_grid3, voxel_grid4]), axis=0).astype(int)

# NN model load
date = "2023_08_26_11_34_47/"
model_file_name = "loss_0.770163893699646_lat512_rnd0_checkpoint_04_512.0000_0_grid.pkl"

model_dir = "model/checkpoints/grid/" + date + model_file_name
param_dir = "model/grid/" + date + "model_param.pickle"

with open(param_dir, "rb") as f:
    NN_param = pickle.load(f)
device = torch.device('cpu')
if NN_param["input_data_shape"]["channel"] == 1:
    assert NN_param["input_data_shape"]["shape"] == [num_grid, num_grid, num_grid]

model = CollNet(
    input_data_shape=NN_param["input_data_shape"] ,
    encoder_layer=NN_param["encoder_layer"],
    decoder_layer=NN_param["decoder_layer"],
    fc_layer_sizes=NN_param["fc_layer"],
    latent_size=NN_param["latent_size"],
    batch_size=1,
    device=device).to(device)

model_state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(model_state_dict)

time.sleep(1)
pc.display(panda_joint_init)
time.sleep(1)

ax1 = plt.figure(1).add_subplot(221)
ax1.set_title("depth image1", fontsize=16, fontweight='bold', pad=20)
ax1.imshow(depth1)
ax2 = plt.figure(1).add_subplot(222)
ax2.set_title("depth image2", fontsize=16, fontweight='bold', pad=20)
ax2.imshow(depth2)
ax3 = plt.figure(1).add_subplot(223)
ax3.set_title("depth image3", fontsize=16, fontweight='bold', pad=20)
ax3.imshow(depth3)
ax4 = plt.figure(1).add_subplot(224)
ax4.set_title("depth image4", fontsize=16, fontweight='bold', pad=20)
ax4.imshow(depth4)

ax9 = plt.figure(2).add_subplot(projection='3d')
ax9.voxels(voxel_grids)
ax9.set_title("voxel grid all", fontsize=16, fontweight='bold', pad=20)
plt.show()

plt.ion()
fig, axs = plt.subplots(num_link, 1, figsize=(6, 2*num_link))
lines1 = []
lines2 = []

for ax in axs:
    line1, = ax.plot([],[], label='ans', color="blue", linewidth=4.0, linestyle='--')
    line2, = ax.plot([],[], label='pred', color = "red", linewidth=2.0)
    ax.legend()
    ax.set_ylim([-0.1,1.1])
    ax.grid()
    lines1.append(line1)
    lines2.append(line2)


def plt_func(fig, lines1, lines2, x_data, y_data, y_hat_data):
    if x_data.shape[0] > 10:
        x_data = x_data[-10:]
        y_data = y_data[-10:]
        y_hat_data = y_hat_data[-10:]
    for link, (line1, line2, ax) in enumerate(zip(lines1, lines2, axs)):
        line1.set_data(x_data, y_data[:, link])
        line2.set_data(x_data, y_hat_data[:, link])
        ax.set_xlim(x_data[0], x_data[-1])
    fig.canvas.draw()
    fig.canvas.flush_events()

x_data = np.zeros((1,1))
y_data = np.zeros((1, num_link))
y_hat_data = np.zeros((1, num_link))



joint_state = panda_joint_init
i=0
for iter in range(1,100000):
    # joint_state = joint_state + np.array([0.0, 0.0, 0.0, -0.002, 0.0, 0.0, 0.0])
    joint_state = joint_limit[0] + np.random.random(7) *( joint_limit[1] - joint_limit[0] )
    pc.display(joint_state)
    coll, dist = pc.collision_vector()
    print(dist)
    if dist != 0:
        continue
    with torch.no_grad():
        model.eval()
        nerf_state = np.concatenate([joint_state, np.cos(joint_state), np.sin(joint_state)],axis=0).astype(np.float32)
        voxel_grids = voxel_grids.astype(np.float32)
        NN_output,_,_,_,_ = model(torch.from_numpy(nerf_state.reshape(1, -1)).to(device),
                                  torch.from_numpy(voxel_grids.reshape(1, 1, num_grid, num_grid, num_grid)).to(device))
    coll_pred = NN_output.cpu().detach().numpy()[0]
    coll.astype(np.int64)
    

    x_data = np.append(x_data, np.array([[i]]), axis=0)
    y_data = np.append(y_data, coll.reshape(1, num_link), axis=0)
    y_hat_data = np.append(y_hat_data, coll_pred.reshape(1, num_link), axis=0)
    

    plt_func(fig, lines1, lines2, x_data, y_data, y_hat_data)
    
    print("=================================")
    print(coll)
    print(coll_pred)
    pc.print_current_collision_infos()
    print("=================================")

    if not np.array_equal(coll, coll_pred):
        plt.pause(4.5)    
    plt.pause(0.5)

    i+=1
plt.show()