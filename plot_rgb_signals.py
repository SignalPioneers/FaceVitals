
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load RGB measurements
data = pd.read_csv("rgb_signals.csv")

# 2. Check whether the CSV contains data
if data.empty:
    print("No RGB measurements found!")
    exit()

# 3. Extract time
time = data["Time"]

# 4. Create three plots
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# ---------------------------------
# FOREHEAD RGB SIGNALS
# ---------------------------------
axes[0].plot(time, data["Forehead_R"],
             color="red", label="Red")

axes[0].plot(time, data["Forehead_G"],
             color="green", label="Green")

axes[0].plot(time, data["Forehead_B"],
             color="blue", label="Blue")

axes[0].set_title("Forehead RGB Signals")
axes[0].set_ylabel("Intensity")
axes[0].legend()
axes[0].grid(True)

# ---------------------------------
# LEFT CHEEK RGB SIGNALS
# ---------------------------------
axes[1].plot(time, data["LeftCheek_R"],
             color="red", label="Red")

axes[1].plot(time, data["LeftCheek_G"],
             color="green", label="Green")

axes[1].plot(time, data["LeftCheek_B"],
             color="blue", label="Blue")

axes[1].set_title("Left Cheek RGB Signals")
axes[1].set_ylabel("Intensity")
axes[1].legend()
axes[1].grid(True)

# ---------------------------------
# RIGHT CHEEK RGB SIGNALS
# ---------------------------------
axes[2].plot(time, data["RightCheek_R"],
             color="red", label="Red")

axes[2].plot(time, data["RightCheek_G"],
             color="green", label="Green")

axes[2].plot(time, data["RightCheek_B"],
             color="blue", label="Blue")

axes[2].set_title("Right Cheek RGB Signals")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Intensity")
axes[2].legend()
axes[2].grid(True)

# 5. Adjust layout
plt.tight_layout()

# 6. Display graphs
plt.show()

# 7. Display recording information
print("Total measurements:", len(data))

duration = time.iloc[-1] - time.iloc[0]

print("Recording duration:",
      round(duration, 2), "seconds")
