import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load both CSVs
original_df = pd.read_csv("/home/arrafi/le-pavd/deep_dynamics/model/DeepDynamicsIAC.csv")
new_df = pd.read_csv("/home/arrafi/le-pavd/deep_dynamics/model/lePAVD_iac.csv")

# Align on minimum length
min_len = min(len(original_df), len(new_df))
original_df = original_df.iloc[:min_len]
new_df = new_df.iloc[3:]



original_df = original_df.reset_index(drop=True)
new_df = new_df.reset_index(drop=True)
# Time conversion (25 Hz sampling)
start_idx = 0
end_idx = min_len
time_axis = np.arange(start_idx, end_idx) / 25.0  # seconds
print(f"Original DataFrame length: {len(original_df)}")
print(f"New DataFrame length: {len(new_df)}")

# Compute Differences
diff_df = pd.DataFrame({
    'ΔFrx': original_df['Frx'] - new_df['Frx'],
    'ΔFry': original_df['Fry'] - new_df['Fry'],
    'ΔFfy': original_df['Ffy'] - new_df['Ffy'],
    'ΔVx':  original_df['Vx']  - new_df['Vx'],
    'ΔVy':  original_df['Vy']  - new_df['Vy'],
    'Δyaw_rate': original_df['yaw_rate'] - new_df['yaw_rate']
})


import matplotlib.pyplot as plt

# Plot Forces Differences with larger fonts
plt.figure(figsize=(14, 10))
plt.rcParams['pdf.fonttype'] = 42  # TrueType
plt.rcParams['ps.fonttype'] = 42
for i, (col, title, color) in enumerate([
    ("ΔFrx", r"Longitudinal Force $F_{rx}$", 'blue'),
    ("ΔFry", r"Rear Lateral Force $F_{ry}$", 'green'),
    ("ΔFfy", r"Front Lateral Force $F_{fy}$", 'red')
]): 
    plt.subplot(3, 1, i + 1)
    
    plt.plot(time_axis, original_df[col[1:]], label=f"DDM {col[1:]}", alpha=1, color='red')
    plt.plot(time_axis, new_df[col[1:]], label=f"eDVDM {col[1:]}", alpha=1, color='green')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.ylabel(title[-9:]+" (N)", fontsize=30)
    
    plt.yticks(fontsize=20)
    plt.xlim(0, 400)
    # if i == 0:
    #     plt.xlim(5550, 6000)
    #     plt.ylim(-500, 2500)
    #     plt.tick_params(labelbottom=False)
    # elif i == 1:
    #     plt.xlim(5550, 6000)
    #     plt.ylim(-500, 500)
    #     plt.tick_params(labelbottom=False)
    # elif i == 2:
    #     plt.xlim(5550, 6000)
    #     plt.ylim(-500, 500)
    #     plt.xlabel("Sample Index", fontsize=24)
    #     plt.xticks(fontsize=20)
    # Only show x-label on the last subplot
    if i < 2:
        plt.tick_params(labelbottom=False)  # hide x labels
    else:
        plt.xlabel("Time (s)", fontsize=30)
        plt.xticks(fontsize=20)
    plt.grid(True)
    
# Shared xlabel
# axes[-1].set_xlabel("Sample Index", labelpad=10, fontsize=24)
plt.xlabel("Time (s)", fontsize=30)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Global font settings
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 28,
    "figure.titlesize": 24
})

plt.rcParams['pdf.fonttype'] = 42  # TrueType
plt.rcParams['ps.fonttype'] = 42
# Create figure and axes
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

state_info = [
    ("ΔVx", r"$V_{x}$ (m/s)", (0, 0)),
    ("ΔVy", r"$V_{y}$ (m/s)", (-0.1, 0.1)),
    ("Δyaw_rate", r"$\omega$ (rad/s)", (-0.1, 0.1))
]

# Plot each subplot
for i, (col, ylabel, ylim) in enumerate(state_info):
    ax = axes[i]
    ax.plot(original_df[col[1:]], label="DDM", color='red')
    ax.plot(new_df[col[1:]], label="eDVDM", color='green')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.75)
    ax.set_ylabel(ylabel, labelpad=20)
    # ax.set_ylim(ylim)
    ax.set_xlim(5550, 6000)
    ax.grid(True)
    ax.tick_params(axis='x', pad=10)
    
    # if i == 0:
    #     ax.set_xlim(5550, 6000)
    #     ax.set_ylim(15, 21)  # Adjust y-limits for better visibility
    # if i == 1:
    #     ax.set_xlim(5550, 6000)
    #     ax.set_ylim(-.05, 0.05)  # Adjust y-limits for better visibility
    # if i == 2:
    #     ax.set_xlim(5550, 6000)
    #     ax.set_ylim(-.05, 0.05)  # Adjust y-limits for better visibility

axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# Shared xlabel
axes[-1].set_xlabel("Sample Index", labelpad=10, fontsize=24)

# Align y-axis labels
fig.align_ylabels(axes)

# Give extra space for y-labels
fig.subplots_adjust(left=0.15, hspace=0.3)

plt.tight_layout()
plt.show()
