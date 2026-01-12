#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ===================== Paths =====================
closed_loop_csv = os.path.expanduser("~/lepavd_closed_loop.csv")
ground_truth_csv = os.path.expanduser("~/lepavd_training_data.csv")

out_dir = os.path.expanduser("~/RESEARCH")
os.makedirs(out_dir, exist_ok=True)

dt = 0.1  # MUST match control dt

# ===================== Load CSV =====================
def load_7d_csv(path):
    data = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

closed = load_7d_csv(closed_loop_csv)
gt = load_7d_csv(ground_truth_csv)

# Match lengths
N = min(len(closed), len(gt))
closed = closed[:N]
gt = gt[:N]

# ===================== Extract Signals =====================
vx_cl, vy_cl, yaw_cl = closed[:, 0], closed[:, 1], closed[:, 2]
vx_gt, vy_gt, yaw_gt = gt[:, 0], gt[:, 1], gt[:, 2]

t = np.arange(N) * dt

# ===================== Integrate velocities to positions =====================
def integrate_position(vx, vy, dt):
    x = np.zeros(len(vx))
    y = np.zeros(len(vy))
    for k in range(1, len(vx)):
        x[k] = x[k - 1] + vx[k - 1] * dt
        y[k] = y[k - 1] + vy[k - 1] * dt
    return x, y

x_cl, y_cl = integrate_position(vx_cl, vy_cl, dt)
x_gt, y_gt = integrate_position(vx_gt, vy_gt, dt)

# ===================== Plot 1: Trajectory =====================
plt.figure()
plt.plot(x_gt, y_gt, 'k--', label="Ground Truth")
plt.plot(x_cl, y_cl, 'r', label="LE-PAVD Closed Loop")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trajectories")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "trajectory_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()


# ===================== Plot 2: Longitudinal Velocity =====================
plt.figure()
plt.plot(t, vx_gt, 'k--', label="Ground Truth $v_x$")
plt.plot(t, vx_cl, 'r', label="Predicted $v_x$")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Longitudinal Velocity Comparison")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "vx_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# ===================== Plot 3: Lateral Velocity =====================
plt.figure()
plt.plot(t, vy_gt, 'k--', label="Ground Truth $v_y$")
plt.plot(t, vy_cl, 'r', label="Predicted $v_y$")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Lateral Velocity Comparison")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out_dir, "vy_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# ===================== Plot 4: Yaw Rate =====================
plt.figure()
plt.plot(t, yaw_gt, 'k--', label="Ground Truth $\dot{\psi}$")
plt.plot(t, yaw_cl, 'r', label="Predicted $\dot{\psi}$")
plt.xlabel("Time [s]")
plt.ylabel("Yaw Rate [rad/s]")
plt.title("Yaw Rate Comparison")
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(out_dir, "yaw_rate_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
