# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import trimesh
from PIL import Image


def visualize_mesh(f, output, show = False, zoom = 1, edges = False, model="dragon"):
    args = {} if not edges else {'color': "none", "edgecolor": "k", "linewidth": 0.1}

    fig = plt.figure(figsize=(10 * zoom, 8 * zoom))
    ax = fig.add_subplot(projection='3d')

    if "dragon" == model:  # ?
        ax.set_proj_type('persp')
    else:
        ax.set_aspect('equal', 'box')

    ax.set_axis_off()

    # Load a trimesh object
    mesh = trimesh.load(f)

    # Compute face normals
    mesh_normals = mesh.face_normals

    # Map normals to colors using a colormap
    norm = Normalize()

    # Assuming you are interested in the z-component of the normals
    if model in ["Armadillo"]:
        colors = plt.cm.plasma(norm(-mesh_normals[:, 2]))
    elif model in ["lucy"]:
        colors = plt.cm.plasma(norm(mesh_normals[:, 1]))
    else:
        colors = plt.cm.plasma(norm(mesh_normals[:, 2]))

    # # Plot the mesh using trimesh's built-in plotting
    # mesh.visual.face_colors = colors
    # mesh.show()

    if model in ["dragon", "lucy"]:
        collec = ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1],
                                 mesh.vertices[:, 2], triangles=mesh.faces, **args)
    else:
        collec = ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 2],
                                 mesh.vertices[:, 1], triangles=mesh.faces, **args)
    collec.set_fc(colors)
    collec.set_edgecolor(colors)
    if "dragon" == model:
        ax.view_init(elev=15, azim=-10, roll=10, vertical_axis="y")
    elif model in ["Armadillo"]:
        ax.view_init(elev=15, azim=-60)
    elif model in ["happy", "drill"]:
        ax.view_init(elev=10, azim=120)
    elif model in ["lucy"]:
        ax.view_init(elev=5, azim=110)
    else:
        ax.view_init(elev=15, azim=120)

    # Save the plot with a transparent background
    plt.savefig(os.path.join(os.path.dirname(__file__), output), transparent=True)

    if show:
        plt.show()

def crop_and_save(input_path, output_path, crop_box):
    # Open the PNG file
    original_image = Image.open(input_path)

    # Crop the specific area defined by crop_box
    cropped_image = original_image.crop(crop_box)

    # Save the cropped image
    cropped_image.save(output_path)

def detect_close_vertices(mesh, threshold_distance):
    # Get the vertices of the mesh
    vertices = mesh.vertices

    # Compute pairwise distances between all vertices
    distances = np.linalg.norm(vertices[:, None] - vertices, axis=-1)

    # Set diagonal elements (distances to itself) to a large value
    np.fill_diagonal(distances, np.inf)

    # Find pairs of vertices that are closer than the threshold distance
    close_vertex_pairs = np.where(distances < threshold_distance)

    # Return the indices of the close vertex pairs
    return list(zip(close_vertex_pairs[0], close_vertex_pairs[1]))


def get_crop_box(data, zoom):
    if "Armadillo" == data:
        dx = -40
        crop_box = [zoom*x for x in (340+dx, 200, 340+370+dx, 200+420)]
    else:
        crop_box = [zoom*x for x in (340, 200, 340+370, 200+420)]
    return crop_box


def visualize_all(seed: int, data: str, size: str, root: str):
    mesh = trimesh.load(f"{root}/../../meshes/{data}/our_mesh_{size}_{seed}.ply")
    zoom = 4

    # threshold_distance = 1e-3
    # close_vertex_pairs = detect_close_vertices(mesh, threshold_distance)
    # print("Close vertex pairs:", len(close_vertex_pairs))

    os.makedirs(f"{root}/outputs/{data}", exist_ok=True)

    visualize_mesh(f"{root}/../../meshes/{data}/our_mesh_{size}_{seed}.ply",
                   f"{root}/outputs/{data}/{size}_ours_{zoom}x.png",
                   show = False, zoom = zoom, edges = True, model=data)
    crop_box = get_crop_box(data, zoom)

    if data in ["bunny", "dragon"]:
        crop_and_save(f"{root}/outputs/{data}/{size}_ours_{zoom}x.png",
                      f"{root}/outputs/{data}/{size}_ours_{zoom}x.png", crop_box)
    else:
        print("warning: no crop information!")

    crop_box = get_crop_box(data, 1)
    visualize_mesh(f"{root}/../../meshes/{data}/our_mesh_{size}_{seed}.ply",
                   f"{root}/outputs/{data}/{size}_ours.png")
    crop_and_save(f"{root}/outputs/{data}/{size}_ours.png",
                  f"{root}/outputs/{data}/{size}_ours.png", crop_box)

    for i in [512, 16, 24, 32, 40, 48, 56, 64, 128, 192, 224, 256]:
        visualize_mesh(f"{root}/../../meshes/{data}/mc{i:03d}_mesh_{size}_{seed}.ply",
                       f"{root}/outputs/{data}/{size}_mc{i:03d}.png")
        crop_and_save(f"{root}/outputs/{data}/{size}_mc{i:03d}.png",
                      f"{root}/outputs/{data}/{size}_mc{i:03d}.png", crop_box)

def visualize_comparison(seed: int, data: str, size: str, root: str):
    mesh = trimesh.load(f"{root}/../../meshes/{data}/our_mesh_{size}_{seed}.ply")
    zoom = 4
    os.makedirs(f"{root}/outputs/{data}", exist_ok=True)

    crop_box = get_crop_box(data, zoom)
    visualize_mesh(f"{root}/../../meshes/{data}/our_mesh_{size}_{seed}.ply",
                   f"{root}/outputs/{data}/{size}_ours_{zoom}x.png",
                   show = False, zoom = zoom, edges = True, model=data)

    if data in ["bunny", "dragon"]:
        crop_and_save(f"{root}/outputs/{data}/{size}_ours_{zoom}x.png",
                      f"{root}/outputs/{data}/{size}_ours_{zoom}x.png", crop_box)
    else:
        print("warning: no crop information!")

    for i in [256]:
        visualize_mesh(f"{root}/../../meshes/{data}/mc{i:03d}_mesh_{size}_{seed}.ply",
                       f"{root}/outputs/{data}/{size}_mc{i:03d}_{zoom}x.png",
                       show = False, zoom = 4, edges = True, model=data)
        crop_and_save(f"{root}/outputs/{data}/{size}_mc{i:03d}_{zoom}x.png",
                      f"{root}/outputs/{data}/{size}_mc{i:03d}_{zoom}x.png", crop_box)

if "__main__" == __name__:
    root = os.path.dirname(os.path.abspath(__file__))

    # Visualizations for the Appendix
    # visualize_all(31, "bunny", "large", root)

    # Detailed comparison shots
    visualize_comparison(35, "bunny", "large", root)
