import numpy as np
import matplotlib.pyplot as plt
import glob
import os

N = 400  # mismo N que usaste en el ejecutable
frame_pattern = "frames/frame_*.bin"

files = sorted(glob.glob(frame_pattern))

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((N, N), dtype=np.float32),
               vmin=0.0, vmax=100.0, cmap="inferno")
plt.colorbar(im, ax=ax)

def load_frame(fname):
    data = np.fromfile(fname, dtype=np.float32)
    return data.reshape((N, N))

def update(frame_idx):
    fname = files[frame_idx]
    grid = load_frame(fname)
    im.set_data(grid)
    ax.set_title(os.path.basename(fname))
    return [im]

# Animación simple
from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=len(files), interval=50, blit=True)
plt.show()
