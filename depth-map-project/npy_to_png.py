import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
npy_file = np.load('data/rs_depth/data_0007.npy')

# Normalize to 0 - 1
npy_file = (npy_file - npy_file.min()) / (npy_file.max() - npy_file.min())

# Apply the plasma colormap
plasma_img = plt.get_cmap('viridis')(npy_file)

# Save the image
plt.imsave('output_image.png', plasma_img)