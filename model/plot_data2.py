import pandas as pd
import matplotlib.pyplot as plt

# Load both CSVs
original_df = pd.read_csv("/home/arrafi/le-pavd/deep_dynamics/model/DeepDynamicsIAC.csv")
new_df = pd.read_csv("/home/arrafi/le-pavd/deep_dynamics/model/lePAVD_iac.csv")

# Align sample lengths
min_len = min(len(original_df), len(new_df))
original_df = original_df.iloc[:min_len]
new_df = new_df.iloc[:min_len]

# Define plot configurations: (force, state, title, colors)
plots = [
    ("Frx", "Vx", "Longitudinal Force vs Velocity", ['blue', 'purple']),
    ("Ffy", "yaw_rate", "Front Lateral Force vs Yaw Rate", ['red', 'brown']),
    ("Fry", "Vy", "Rear Lateral Force vs Lateral Velocity", ['green', 'orange'])
]

plt.figure(figsize=(14, 12))
for i, (force, state, title, colors) in enumerate(plots):
    plt.subplot(3, 1, i + 1)

    # Plot original model (dashed lines)
    plt.plot(original_df[force], label=f"Original {force}", linestyle='--', color=colors[0], alpha=0.7)
    plt.plot(original_df[state], label=f"Original {state}", linestyle='--', color=colors[1], alpha=0.7)

    # Plot enhanced model (solid lines)
    plt.plot(new_df[force], label=f"New {force}", linestyle='-', color=colors[0])
    plt.plot(new_df[state], label=f"New {state}", linestyle='-', color=colors[1])

    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
