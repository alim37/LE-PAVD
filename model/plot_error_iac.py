import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import matplotlib.animation as animation 
import csv
import os
import yaml
import torch
import pickle
from tqdm import tqdm
from models import string_to_model, string_to_dataset
from models_lePAVD import string_to_lePAVD, string_to_LEdataset
from csv_parser import write_dataset

Ts = 0.04
HORIZON = 15


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project root:", project_root)

#####################################################################
# load track
TRACK_NAME = "lvms"
inner_bounds = []
with open(project_root+ "/tracks/" + TRACK_NAME + "_inner_bound.csv") as f:
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		inner_bounds.append([float(row[0]), float(row[1])])
outer_bounds = []


with open(project_root+ "/tracks/" + TRACK_NAME + "_outer_bound.csv") as f:
	reader = csv.reader(f, delimiter=',')
 
	for row in reader:
		outer_bounds.append([float(row[0]), float(row[1])])
inner_bounds = np.array(inner_bounds, dtype=np.float32)
outer_bounds = np.array(outer_bounds, dtype=np.float32)
#####################################################################
# load inputs used to simulate Dynamic model

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")



# dataset_file = os.path.join(project_root, "data", "combined_Putnam_park2023.csv")
# dataset_file = os.path.join(project_root, "data", "combined_LVMS_23_01_04.csv")

# dataset_file = os.path.join(project_root, "data", "LVMS_23_01_04_A.csv")
dataset_file = os.path.join(project_root, "data", "LVMS_23_01_04_B.csv")
# dataset_file = os.path.join(project_root, "data", "combined_LVMS_23_01_04.csv")

param_file = os.path.join(project_root, "cfgs", "model", "lePAVD_iac.yaml")
state_dict = os.path.join(project_root, "output", "lePAVD_iac", "lePAVD_iac_putnam", "epoch_3490.pth")
scaler_path = os.path.join(project_root, "output", "lePAVD_iac", "lePAVD_iac_putnam", "scaler.pkl")


with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
with open(os.path.join(os.path.dirname(state_dict), "scaler.pkl"), "rb") as f:
	eddm_scaler = pickle.load(f)
eddm = string_to_lePAVD[param_dict["MODEL"]["NAME"]](param_dict, eval_mode=False)
eddm.to(device)
eddm.eval()
eddm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, eddm.horizon, save=False)
stop_idx = len(poses) + eddm.horizon

samples = list(range(50, 300, 50))
driving_inputs = features[:,0,3:5] + features[:,0,5:7]
eddm_dataset = string_to_LEdataset[param_dict["MODEL"]["NAME"]](features, labels, eddm_scaler)
eddm_predictions = np.zeros((stop_idx, 3))
eddm_data_loader = torch.utils.data.DataLoader(eddm_dataset, batch_size=1, shuffle=False)
idt = 0
states = np.zeros((stop_idx, 3))
for inputs, labels, norm_inputs in tqdm(eddm_data_loader, total=len(eddm_predictions)):
	if idt == len(eddm_predictions):
		break
	if eddm.is_rnn:
		h = eddm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if eddm.is_rnn:
		eddm_state, h, _ = eddm(inputs, norm_inputs, h)
	else:
		eddm_state, _, _ = eddm(inputs, norm_inputs)
	# Simulate model
	eddm_state = eddm_state.cpu().detach().numpy()[0]
	idx = 0
	eddm_predictions[idt+eddm.horizon,:] = eddm_state
	states[idt+eddm.horizon,:] = labels.cpu().numpy()
	idt += 1


# # ddm 

param_file = os.path.join(project_root, "cfgs", "model", "deep_dynamics_iac.yaml")
state_dict = os.path.join(project_root, "output", "deep_dynamics_iac", "ddm_iac_putnam", "epoch_376.pth")
scaler_path = os.path.join(project_root, "output", "deep_dynamics_iac", "ddm_iac_putnam", "scaler.pkl")


with open(param_file, 'rb') as f:
	param_dict = yaml.load(f, Loader=yaml.SafeLoader)
with open(os.path.join(os.path.dirname(state_dict), "scaler.pkl"), "rb") as f:
	ddm_scaler = pickle.load(f)
ddm = string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=False)
ddm.to(device)
ddm.eval()
ddm.load_state_dict(torch.load(state_dict))
features, labels, poses = write_dataset(dataset_file, ddm.horizon, save=False)
stop_idx = len(poses) + ddm.horizon
# for i in range(len(poses)): ## Odometry set to 0 when lap is finished
# 	if poses[i,0] == 0.0 and poses[i,1] == 0.0:
# 		stop_idx = i
# 		break
samples = list(range(50, 300, 50))
driving_inputs = features[:,0,3:5] + features[:,0,5:7]
ddm_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](features, labels, ddm_scaler)
ddm_predictions = np.zeros((stop_idx, 3))
ddm_data_loader = torch.utils.data.DataLoader(ddm_dataset, batch_size=1, shuffle=False)
idt = 0
states = np.zeros((stop_idx, 3))
for inputs, labels, norm_inputs in tqdm(ddm_data_loader, total=len(ddm_predictions)):
	if idt == len(ddm_predictions):
		break
	if ddm.is_rnn:
		h = ddm.init_hidden(inputs.shape[0])
		h = h.data
	inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
	if ddm.is_rnn:
		ddm_state, h, _ = ddm(inputs, norm_inputs, h)
	else:
		ddm_state, _, _ = ddm(inputs, norm_inputs)
	# Simulate model
	ddm_state = ddm_state.cpu().detach().numpy()[0]
	idx = 0
	ddm_predictions[idt+ddm.horizon,:] = ddm_state
	states[idt+ddm.horizon,:] = labels.cpu().numpy()
	idt += 1



import matplotlib.pyplot as plt
import numpy as np

# Ensure predictions and ground truth are aligned
time = np.array(range(max(len(eddm_predictions), len(ddm_predictions)))) * Ts
start_idx = max(eddm.horizon, ddm.horizon)
time = time[start_idx:-20]
eddm_predictions = eddm_predictions[start_idx:-17, :]
ddm_predictions = ddm_predictions[start_idx:-20, :]
states = states[start_idx:-20, :]

print("Vx", states[:, 0].mean(), ddm_predictions[:, 0].mean(), eddm_predictions[:, 0].mean())
print("Vy", states[:, 1].mean(), ddm_predictions[:, 1].mean(), eddm_predictions[:, 1].mean())
print("W", states[:, 2].mean(), ddm_predictions[:, 2].mean(), eddm_predictions[:, 2].mean())


print(eddm_predictions.shape, ddm_predictions.shape, states.shape)



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Global font settings for LaTeX figures
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 28,
    "figure.titlesize": 24
})

plt.rcParams['pdf.fonttype'] = 42  # TrueType
plt.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# --------- vx Plot ---------
fig_vx, ax_vx = plt.subplots(figsize=(10, 4))
ax_vx.plot(time, states[:, 0], 'b', label='Ground Truth')
ax_vx.plot(time, ddm_predictions[:, 0], 'r', label='DDM')
ax_vx.plot(time, eddm_predictions[:, 0], 'g', label='eDVDM')
ax_vx.set_ylabel(r"$v_x$ (m/s)")
ax_vx.set_xlabel("Time (s)")
ax_vx.grid(True)
# ax_vx.legend()

# Zoom-in inset for vx
axins_vx = inset_axes(ax_vx, width="40%", height="45%", loc='lower right', borderpad=1)
x1, x2 = 110, 120
y1, y2 = 17.8, 18.0
axins_vx.plot(time, states[:, 0], 'b')
axins_vx.plot(time, ddm_predictions[:, 0], 'r')
axins_vx.plot(time, eddm_predictions[:, 0], 'g')
axins_vx.set_xlim(x1, x2)
axins_vx.set_ylim(y1, y2)
axins_vx.set_xticks([])
axins_vx.set_yticks([])
mark_inset(ax_vx, axins_vx, loc1=2, loc2=4, fc="none", ec="green", linestyle="--")


# --------- vy Plot ---------
fig_vy, ax_vy = plt.subplots(figsize=(10, 4))
ax_vy.plot(time, states[:, 1], 'b', label='Ground Truth')
ax_vy.plot(time, ddm_predictions[:, 1], 'r', label='DDM')
ax_vy.plot(time, eddm_predictions[:, 1], 'g', label='eDVDM')
ax_vy.set_ylabel(r"$v_y$ (m/s)")
ax_vy.set_xlabel("Time (s)")
ax_vy.grid(True)
# ax_vy.legend()

# Zoom-in inset for vy
axins_vy = inset_axes(ax_vy, width="40%", height="45%", loc='upper right', borderpad=1)
x1, x2 = 100, 130
y1, y2 = -0.07, 0.2
axins_vy.plot(time, states[:, 1], 'b')
axins_vy.plot(time, ddm_predictions[:, 1], 'r')
axins_vy.plot(time, eddm_predictions[:, 1], 'g')
axins_vy.set_xlim(x1, x2)
axins_vy.set_ylim(y1, y2)
axins_vy.set_xticks([])
axins_vy.set_yticks([])
mark_inset(ax_vy, axins_vy, loc1=2, loc2=4, fc="none", ec="green", linestyle="--")


# --------- omega Plot ---------
fig_omega, ax_omega = plt.subplots(figsize=(10, 4))
ax_omega.plot(time, states[:, 2], 'b', label='Ground Truth')
ax_omega.plot(time, ddm_predictions[:, 2], 'r', label='DDM')
ax_omega.plot(time, eddm_predictions[:, 2], 'g', label='eDVDM')
ax_omega.set_ylabel(r"$\omega$ (rad/s)")
ax_omega.set_xlabel("Time (s)")
ax_omega.grid(True)
# ax_omega.legend()

# Zoom-in inset for omega
axins_omega = inset_axes(ax_omega, width="40%", height="45%", loc='upper right', borderpad=1)
x1, x2 = 80, 110
y1, y2 = -0.03, 0.03
axins_omega.plot(time, states[:, 2], 'b')
axins_omega.plot(time, ddm_predictions[:, 2], 'r')
axins_omega.plot(time, eddm_predictions[:, 2], 'g')
axins_omega.set_xlim(x1, x2)
axins_omega.set_ylim(y1, y2)
axins_omega.set_xticks([])
axins_omega.set_yticks([])
mark_inset(ax_omega, axins_omega, loc1=2, loc2=4, fc="none", ec="green", linestyle="--")
# plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()

plt.rcParams['pdf.fonttype'] = 42  # TrueType
plt.rcParams['ps.fonttype'] = 42

# Plot absolute errors
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

error_vx = np.abs(states[:, 0] - ddm_predictions[:, 0])
error_vx_e = np.abs(states[:, 0] - eddm_predictions[:, 0])

error_vy = np.abs(states[:, 1] - ddm_predictions[:, 1])
error_vy_e = np.abs(states[:, 1] - eddm_predictions[:, 1])

error_w = np.abs(states[:, 2] - ddm_predictions[:, 2])
error_w_e = np.abs(states[:, 2] - eddm_predictions[:, 2])

# vx error
# ax[0].plot(time, states[:, 0], 'b', label='Ground Truth')
ax[0].plot(time, error_vx, 'r', label='DDM')
ax[0].plot(time, error_vx_e, 'g', label='eDVDM')
ax[0].set_ylabel(r"$\epsilon_{v_x}$ (m/s)")
# ax[0].legend()
ax[0].grid(True)

# vy error
ax[1].plot(time, error_vy, 'r')
ax[1].plot(time, error_vy_e, 'g')
ax[1].set_ylabel(r"$\epsilon_{v_y}$ (m/s)")
ax[1].grid(True)

# yaw rate error
ax[2].plot(time, error_w, 'r')
ax[2].plot(time, error_w_e, 'g')
ax[2].set_ylabel(r"$\epsilon_{\omega}$ (rad/s)")
ax[2].set_xlabel("Time (s)", fontsize=24)
ax[2].grid(True)

plt.xlim(0, 400) 

# fig.suptitle("Comparison of Absolute State Errors Over Time", fontsize=30)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
