# Copyright (C) 2022-present Naver Corporation / Inria centre at the University Grenoble Alpes. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import argparse
import trimesh
import random
from ipdb import set_trace  as bb
import numpy as np
osp = os.path
import matplotlib.pyplot as plt
from dataset_info import  SHOWMeDatasetInterface
import matplotlib.image as mpimg
from PIL import Image


intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
            )

def project(P, X):
    """
    X: Nx3
    P: 3x4 projection matrix
    returns Nx2 perspective projections
    """
    X = np.vstack((X.T, np.ones(len(X))))
    x = P @ X
    x = x[:2] / x[2]
    return x.T


def showimgs_sidebyside(rgb_image, depth_image, title=None, savepth=None):
    """
    Visualize RGB and Depth images side by side.
    Args:
    - rgb_image (numpy.ndarray): RGB Image
    - depth_image (numpy.ndarray): Depth Image (grayscale)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.set_title('RGB Image')
    ax1.imshow(rgb_image)
    ax1.axis('off')

    ax2.set_title('Depth Image')
    im2 = ax2.imshow(depth_image, cmap='viridis')
    ax2.axis('off')
    # cbar = plt.colorbar(im2, ax=ax2)
    # cbar.set_label('Depth')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    if savepth is not None:
        plt.savefig(savepth)


def visualize_depth_image(depth_image, title=None, savepth=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title(f"{title} Depth Image Visualization")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    if savepth is not None:
        plt.savefig(savepth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/scratch/2/data/SHOWMe', help='dataset root dir pth') 
    parser.add_argument('--depth_datadir', type=str, default='/scratch/2/data/SHOWMe_depth', help='depth dataset dir pth') 
    parser.add_argument('--outdir', type=str, default='./out', help='output') 
    parser.add_argument('--vis_type', type=str, default=None, help='rgb, depth, homesh, objmesh, manomesh')
    parser.add_argument('--seq_id', type=str, default=None, help='sequence id')
    parser.add_argument('--frm_no', type=int, default=None, help='frame number')
    args = parser.parse_args()

    dset_interf = SHOWMeDatasetInterface(datadir=args.datadir)

    all_seq_ids = dset_interf.get_all_sqns_ids()
    if args.seq_id is None:
        seq_id = random.choice(all_seq_ids)
    else:
        seq_id = args.seq_id

    if args.frm_no is None:
        img_pth = random.choice(dset_interf.get_seq_rgbs_pths(seq_id))
        fname = osp.basename(img_pth)
        dimg_pth = osp.join(args.depth_datadir, seq_id, f"depth/{fname.replace('png', 'xyz.npz')}")
    else:
        img_pth = osp.join(args.datadir, seq_id, f'rgb/{args.frm_no:010d}.png')
        dimg_pth = osp.join(args.depth_datadir, seq_id, f'depth/{args.frm_no:010d}.xyz.npz')

    assert os.path.exists(img_pth), f"{img_pth} does not exist"
    assert os.path.exists(dimg_pth), f"{dimg_pth} does not exist"

    if args.vis_type == 'rgb':
        print(img_pth)
        img = mpimg.imread(img_pth)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif args.vis_type == 'depth':
        dimg = np.load(dimg_pth)['arr_0'][:, :, 2]
        title = f"Frame No.: {int(osp.basename(dimg_pth).split('.')[0])}"
        visualize_depth_image(depth_image=dimg, title=title, savepth=f"{args.outdir}/{int(osp.basename(dimg_pth).split('.')[0])}_depth.png")
    elif args.vis_type == 'rgbd':
        img = mpimg.imread(img_pth)
        dimg = np.load(dimg_pth)['arr_0'][:, :, 2] # only z-axis values are considered depth
        title = f"Frame No.: {int(osp.basename(dimg_pth).split('.')[0])}"
        showimgs_sidebyside(rgb_image=img, depth_image=dimg, title=title, savepth=f"{args.outdir}/{int(osp.basename(dimg_pth).split('.')[0])}_rgbd.png")
    elif args.vis_type == 'ho_mesh':
        gmesh_pth = osp.join(args.datadir, seq_id, f'gt_mesh/ho/homesh.obj')
        mesh = trimesh.load(gmesh_pth)
        mesh.show()
    elif args.vis_type == 'mano_mesh':
        gmesh_pth = osp.join(args.datadir, seq_id, f'gt_mesh/mano_mesh/mano.obj')
        mesh = trimesh.load(gmesh_pth)
        mesh.show()
    elif args.vis_type == 'obj_mesh':
        gmesh_pth = osp.join(args.datadir, seq_id, f'gt_mesh/obj_mesh/objmesh.obj')
        mesh = trimesh.load(gmesh_pth)
        mesh.show()
    elif args.vis_type == 'proj_verts':
        gmesh_pth = osp.join(args.datadir, seq_id, f'gt_mesh/ho/homesh.obj')
        verts = trimesh.load(gmesh_pth).vertices
        img = mpimg.imread(img_pth)
        frm_name = osp.basename(img_pth).split('.')[0]
        pose_pth = osp.join(args.datadir, seq_id, 'icp_res', frm_name, 'f_trans.txt')
        pose = np.linalg.inv(np.loadtxt(pose_pth))
        proj_verts = project(P=(intrinsics @ pose[:3, :]), X=verts)
        plt.imshow(img)
        plt.scatter(proj_verts[:, 0], proj_verts[:, 1], s=1, c='r', alpha=0.02)
        plt.axis('off')
        plt.show()
    elif args.vis_type == 'pix_algnd_depth':
        depth_pts = np.load(dimg_pth)['arr_0'].reshape(-1, 3)
        clrs = np.asarray(Image.open(img_pth)).reshape(-1, 3)
        dpcd = trimesh.PointCloud(vertices=depth_pts, colors=clrs)
        dpcd.show()
    else:
        raise ValueError("Unknown 'vis_type' value")
    
