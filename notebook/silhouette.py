#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import gif
import pylab as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from PIL import ImageColor
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap
from pythonperlin import perlin, extend2d
import time


def _is_int(x, epsilon=1e-5):
    """
    Tests if argument is integer

    """
    if isinstance(x, np.ndarray) or isinstance(x, list):
        mask = np.all([is_int(x_) for x_ in x])
    else:
        mask = np.abs(x - np.round(x)) < epsilon
    return mask


def remove_margins():
    """
    Removes figure margins, keeps only plot area 
    """
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    return


def colors_to_cmap(colors):
    """
    Converts list of colors to matplotlib colormap
    """
    cmap = LinearSegmentedColormap.from_list("cmap", colors)
    return cmap


def load_image(fname):
    image = Image.open(os.path.expanduser(fname))
    # Convert to NumPy array
    img = image.convert("RGB")
    img = np.array(img).astype(float)
    # Convert to Black & White (0 and 1, respectively)
    img = np.sum(img, axis=2).T[:,::-1] / (3 * 255)
    img = (img > 0.5).astype(float)
    return img


def split_into_blocks(x, nx, ny):
    """
    Splits an array into nx 8 ny blocks
    """
    blocks = np.hsplit(x, ny)
    blocks = np.vstack(blocks)
    blocks = np.split(blocks, nx * ny)
    return blocks


def com2d(x):
    """
    Calculates COM (center of mass) of points in a 2D array
    """
    assert x.ndim == 2
    com = np.zeros((x.ndim))
    if np.std(x):
        s = np.sum(x)
        for axis in range(x.ndim):
            d = x.shape[axis]
            i = (np.arange(d) + .5) / d - .5
            if axis:
                com[axis] = np.sum(np.dot(x, i), axis=0) / s
            else:
                com[axis] = np.sum(np.dot(x.T, i), axis=0) / s
    return com


def detect_edges_and_fills(x, nx, ny):
    """
    Detects edges and fill in a 2D numpy array of a b&w image
    """
    assert _is_int(x.shape[0]/ nx) and _is_int(x.shape[1] / ny)
    blocks = split_into_blocks(x, nx, ny)
    dx = []
    dy = []
    dz = []
    for b in blocks:
        # Edges: Calc difference between black & white centers of mass
        d = com2d(1 - b) - com2d(b)
        dx.append(d[0])
        dy.append(d[1])
        # Fill: Calc fraction of black points in block
        dz.append(1 - np.sum(b) / b.size)
    dx = np.array(dx).reshape(ny, nx).T
    dy = np.array(dy).reshape(ny, nx).T
    dz = np.array(dz).reshape(ny, nx).T
    return dx, dy, dz


def calc_edge_levels(dx, dy):
    """
    Propagates closeness to edge to all distand grid nodes
    """
    levels = (np.abs(dx) > 0) | (np.abs(dy) > 0)
    levels = levels.astype(float)
    level = np.max(levels) + 1
    start = True
    while start:
        start = False
        l_next = np.copy(levels)
        for i in [-1, 1]:
            for j in [-1, 1]:
                l = np.pad(levels, (1,1))
                l = np.roll(l, i, 0)
                l = np.roll(l, j, 1)
                l = l[1:-1][:,1:-1]
                l_next[l > 0] = 1
        mask = (levels == 0) & (l_next > 0)
        if np.any(mask):
            start = True
            levels[mask] = level
            level += 1
    return levels


def image_to_grid(fname, nx, ny):
    # Load B&W silhouette image as 2D array
    img = load_image(fname)
    # Generate coords of grid nodes
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    # Calc direction to edges and fill fractions at each grid node
    dx, dy, dz = detect_edges_and_fills(img, nx, ny)
    # Calc distance to edges
    z = calc_edge_levels(dx, dy)
    return x, y, z, dx, dy, dz


def colorize(grid, cmaps, scale=1):
    x, y, z, dx, dy, dz = grid
    bg, cmap = cmaps
    nx, ny = x.shape
    c = []
    n = len(x)
    for i in range(nx):
        for j in range(ny):
            u = 0.85 / z[i,j]
            w = 0.75 * (1 - j / ny) 
            w = np.clip(max(scale * u, w), 0, 1)
            c.append(cmap(w))
    c = [c_ for c_ in np.array(c)]
    return c


class ImageToWireframe():
    
    def __init__(self, nx=25, ny=35, nz=8, dens=32, seed=0):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dens = dens
        self.seed = seed
        self.w = 2 * np.pi / (self.nz * self.dens)
        self.cmaps = {
                "cyberpunk": ["#0D0018", colors_to_cmap(["#42007A", "#7D0CA6", "#B719D3", "#F225FF", "#5C9BE8"])],
                "synthwave": ["#0D0018", colors_to_cmap(["#42007A", "#641C73", "#85386D", "#A75466", "#FBD606"])],
                "blue": ["#000022", colors_to_cmap(["#1D90FF", "#3DA0FF", "#5EB0FF", "#7EBFFF", "#7EBFFF"])],
                "green": ["#001111", colors_to_cmap(["#304F4F", "#208A8A", "#10C4C4", "#00FFFF", "#10C4C4"])],
            }

        self.get_noise()
        print(self.px.shape)


    def get_noise(self):
        """
        Load or generate perlin noise
        """
        fname = f"perlin_{self.nx}_{self.ny}.npy"
        try:
            p = np.load(fname).astype(float)
        except:
            t0 = time.time()
            shape = (nz, self.nx, self.ny)
            p = perlin(shape, dens=dens, seed=seed)
            np.save(fname, self.p.astype(np.float16))
            dt = time.time() - t0
            nmin = dt // 60
            nsec = dt % 60
            print(f"Time: {nmin:.0f} min {nsec:.1f} sec")
        d, d2 = self.dens // 2, self.dens
        p = p[:,d:][:,:,d:]
        p = p[:,::d2][:,:,::d2]
        z = np.exp(2j * np.pi * p)
        self.px = z.real
        self.py = z.imag
        return


    def draw_frame(self, i, fname, cmap="cyberpunk"):
        grid = image_to_grid(fname, self.nx, self.ny)
        x, y, z, dx, dy, dz = grid

        # Add noise to x and y
        s = 1.5
        xn = x + s * self.px[i]
        yn = y + s * self.py[i]
        xn = extend2d(xn[:,::2][::2])
        yn = extend2d(yn[:,::2][::2])

        # Add displacement to x and y
        s = 1.5
        xd = x + s * dx
        yd = y + s * dy

        # Periodically mix noisy and displaced x and y
        scale = 1 / np.power(z, 0.25)
        f = ((z <= 3) | (dz > 0)).astype(float)
        periodic = 0.5 * (1 + np.cos(self.w * i + np.pi))
        f = f * scale * (0.3 + 0.6 * periodic)
        x = f * xd + (1 - f) * xn
        y = f * yd + (1 - f) * yn

        # Populate wires along x axis
        q = 4
        x = extend2d(x, q, 0)
        y = extend2d(y, q, 0)
        


        # Draw frame with aspect ratio 0.8 (1080 x 1350)
        q += 1
        color = colorize(grid, self.cmaps[cmap], periodic)
        fig = plt.figure(figsize=(6,7.5), facecolor=self.cmaps[cmap][0])
        remove_margins()
        for j in range(x.shape[0]):
            for k in range(x.shape[1]-1):
                u = self.ny * (j // q) + k
                c = color[u]
                if j % q:
                    v = self.ny * ((j+q) // q) + k
                    f = (j % q) / q
                    c = (1 - f) * color[u] + f * color[v]
                xs = [x[j,k], x[j,k+1]]
                ys = [y[j,k], y[j,k+1] - 0.08]
                plt.plot(xs, ys, color=c)
        plt.xlim(-3, self.nx + 3)
        plt.ylim(-1, self.ny + 1)
        plt.tight_layout()
        return fig


    def draw_animation(self, fname, cmap="cyberpunk", output="wireframe.gif"):
        # Set the dots per inch resolution
        gif.options.matplotlib["dpi"] = 180

        # Decorate a plot function with @gif.frame
        @gif.frame
        def plot(i, fname, cmap):
            self.draw_frame(i, fname, cmap)

        # Construct "frames"
        frames = [plot(i, fname, cmap) for i in range(0, 256, 2)]

        # Save "frames" to gif with a specified duration (milliseconds) between each frame
        gif.save(frames, output, duration=120)











