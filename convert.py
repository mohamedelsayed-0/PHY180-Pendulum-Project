import numpy as np

L = 0.3  # meters

data = np.loadtxt("data3.txt", comments="#")

# Extract time and x
t = data[:, 0]
x = data[:, 1]

# Compute angle in radians
theta_rad = np.arcsin(x / L)

# Stack t and theta column-wise
output_data = np.column_stack((t, theta_rad))

np.savetxt("angles.txt", output_data, header="t theta_rad", fmt="%.6f")

print("angles.txt created successfully with t and theta (radians).")
