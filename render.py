import numpy as np, matplotlib.pyplot as plt
from PIL import Image
import glob, os

with open("frames.bin","rb") as f:
    n = np.fromfile(f, dtype=np.int32, count=1)[0]
    frames = np.fromfile(f, dtype=np.float32).reshape(-1,2,n)

os.makedirs("pngs", exist_ok=True)
L = 2.5
for i in range(len(frames)):
    fig = plt.figure(figsize=(6,6), facecolor="black")
    ax = plt.axes(xlim=(-L,L), ylim=(-L,L)); ax.set_facecolor("black"); ax.axis("off")
    ax.scatter(frames[i,0], frames[i,1], s=0.4, c="white", alpha=0.6)
    fig.savefig(f"pngs/{i:03d}.png", facecolor="black", dpi=60)
    plt.close(fig)

imgs = [Image.open(p) for p in sorted(glob.glob("pngs/*.png"))]
imgs[0].save("galaxy.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
print("saved galaxy.gif from", len(imgs), "frames")
