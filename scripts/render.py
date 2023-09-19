# Copyright (C) 2022-present Naver Corporation / Inria centre at the University Grenoble Alpes. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Reference: https://github.com/naver/posebert/blob/master/renderer.py

import argparse
import torch
import pytorch3d
import pytorch3d.utils
import pytorch3d.renderer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.renderer import look_at_view_transform
from PIL import Image
import os, json
from tqdm import tqdm
from ipdb import set_trace as bb
from pytorch3d.renderer import (
    PointLights,
)
osp = os.path

def trnsfm_points(trnsfm, pts):
    """
    pts: Nx3
    trnsfm: 4x4 homogeneous
    """
    if trnsfm.shape == (3, 4):
        trnsfm_hom = np.vstack([trnsfm, np.array([0, 0, 0, 1])])
    else:
        trnsfm_hom = trnsfm

    pts = np.vstack((pts.T, np.ones(len(pts))))
    pts = trnsfm_hom @ pts
    pts = pts[:3].T

    return pts

def load_mesh(path,filename):
    strFileName = os.path.join(path,filename)
    vertices = []
    texcoords = []
    normals = []
    faces = []
    texInd = []
    texture = None
    
    #first, read file and build arrays of vertices and faces
    for line in open(strFileName, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'mtllib':
            texture = loadMaterialTexture(path+"/"+values[1])
        if values[0] == 'v':
            vertices.append(list(map(float, values[1:4])))
            
        elif values[0] == 'vn':
            v = list(map(float, values[1:4]))
            normals.append(v)
        elif values[0] == 'vt':
            texcoords.append(list(map(float, values[1:3])))
            # continue
        elif values[0] in ('usemtl', 'usemat'):
            continue
        elif values[0] == 'f':
            for triNum in range(len(values)-3):  ## one line fan triangulation to triangulate polygons of size > 3
                v = values[1]
                w = v.split('/')
                faces.append(int(w[0])-1)           #vertex index (1st vertex)
                
                if(len(w)>1):
                    # print(len(w), w)
                    try:
                        texInd.append(int(w[1])-1)
                    except:
                        texInd.append(int(w[2])-1)

                for v in values[triNum+2:triNum+4]:
                    w = v.split('/')
                    faces.append(int(w[0])-1)           #vertex index (additional vertices)
                    if(len(w)>1):
                        try:
                            texInd.append(int(w[1])-1)
                        except:
                            texInd.append(int(w[2])-1)

    
    vertices = np.array(vertices).astype(np.float32)
    texcoords = np.array(texcoords).astype(np.float32)
    nb_vert = vertices.shape[0]

    # If 16 bits are not enough to write vertex indices, use 32 bits 
    if nb_vert<65536:
        faces = np.array(faces).reshape(len(faces) // 3, 3).astype(np.uint16)
    else:
        faces = np.array(faces).reshape(len(faces) // 3, 3).astype(np.uint32)
    if len(texInd)>0:
        if texcoords.shape[0]<65536:
            texInd = np.array(texInd).reshape(faces.shape).astype(np.uint16)
        else:
            texInd = np.array(texInd).reshape(faces.shape).astype(np.uint32)

    return vertices, faces, texcoords, texInd, texture

def loadMaterialTexture(strFileName):

    txtFileName = ''
    for line in open(strFileName, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue

        if values[0] == "newmtl":
            continue
        elif values[0] == "Ka":
            # new_mat.ka = [float(values[1]),float(values[2]),float(values[3])]
            continue
        elif values[0] == "Kd":
            # new_mat.kd = [float(values[1]),float(values[2]),float(values[3])]
            continue
        elif values[0] == "Ks":
            # new_mat.ks = [float(values[1]),float(values[2]),float(values[3])]
            continue
        elif values[0] == "illum":
            # new_mat.illul = values[1]
            continue
        elif values[0] == "map_Ka":
            # txtFileName = getPath(strFileName)+"/"+values[1]
            txtFileName = os.path.dirname(strFileName)+"/"+values[1]
        elif values[0] == "map_Kd":
            # txtFileName = getPath(strFileName)+"/"+values[1]
            txtFileName = os.path.dirname(strFileName)+"/"+values[1]

    myIm = Image.open(txtFileName)
    npim = np.array(myIm)

    return npim

class PyTorch3DRenderer(torch.nn.Module):
    """
    Thin wrapper around pytorch3d threed.
    Only square renderings are supported.
    Remark: PyTorch3D uses a camera convention with z going out of the camera and x pointing left.
    """

    def __init__(self,
                 image_size,
                 background_color=(0, 0, 0),
                 convention='opencv',
                 blur_radius=1e-10,
                 faces_per_pixel=1,
                 bg_blending_radius=1,
                 max_faces_per_bin=200000,
                #  max_faces_per_bin=2000000,
                 # https://github.com/facebookresearch/pytorch3d/issues/448
                 # https://github.com/facebookresearch/pytorch3d/issues/348
                 # https://github.com/facebookresearch/pytorch3d/issues/316
                 ):
        super().__init__()
        self.image_size = image_size
        raster_settings_soft = pytorch3d.renderer.RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=max_faces_per_bin,
            )
        rasterizer = pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings_soft)

        materials = pytorch3d.renderer.materials.Materials(shininess=1.0)
        self.background_color = background_color
        blend_params = pytorch3d.renderer.BlendParams(background_color=background_color,
        # gamma=1e-4,
        # sigma=1e-4,
        # sigma=0.15,
        # gamma=0.14,
        )
        self.blend_params = blend_params
        print('blend_params', blend_params)

        # One need to attribute a camera to the shader, otherwise the method "to" does not work.
        dummy_cameras = pytorch3d.renderer.OrthographicCameras()
        shader = pytorch3d.renderer.SoftPhongShader(cameras=dummy_cameras,
                                                    materials=materials,
                                                    blend_params=blend_params)

        # Differentiable soft threed using per vertex RGB colors for texture
        # self.renderer = pytorch3d.renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)
        self.renderer = pytorch3d.renderer.MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

        self.convention = convention
        if convention == 'opencv':
            # Base camera rotation
            base_rotation = torch.as_tensor([[[-1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, 1]]], dtype=torch.float)
            self.register_buffer("base_rotation", base_rotation)
            self.register_buffer("base_rotation2d", base_rotation[:, 0:2, 0:2])

        # Light Color
        self.ambient_color = 0.5
        self.diffuse_color = 0.3
        self.specular_color = 0.2

        self.bg_blending_radius = bg_blending_radius
        if bg_blending_radius > 0:
            self.register_buffer("bg_blending_kernel",
                                 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (
                                         2 * bg_blending_radius + 1) ** 2)
            self.register_buffer("bg_blending_bias", -torch.ones(1))
        else:
            self.blending_kernel = None
            self.blending_bias = None

    def compose_foreground_on_background(self, fg_img, fg_masks, bg_img, alpha=1.):
        """
        Args:
            - fg_img: [B,3,W,H]
            - fg_mask: [B,W,H]
            - bg_img: [B,3,W,H]
        Copy-paste foreground on a background using the foreground masks.
        Done using a simple smoothing or by hard copy-pasting.
        """

        if self.bg_blending_radius > 0:
            # Simple smoothing of the mask
            fg_masks = torch.clamp_min(
                torch.nn.functional.conv2d(fg_masks.unsqueeze(1), weight=self.bg_blending_kernel, bias=self.bg_blending_bias,
                                           padding=self.bg_blending_radius) * fg_masks.unsqueeze(1), 0.0)[:,0].unsqueeze(-1)
        out = (alpha* fg_img + (1-alpha) * bg_img )* fg_masks+ bg_img * (1.0 - fg_masks)
        return out

    def to(self, device):
        # Transfer to device is a bit bugged in pytorch3d, one needs to do this manually
        self.renderer.shader.to(device)
        return super().to(device)

    def render(self, vertices, faces, cameras, color=None, faces_uvs=None, verts_uvs=None):
        """
        Args:
            - vertices: [B,N,V,3] OR list of shape [V,3]
            - faces: [B,F,3] OR list of shape [F,3]
            - maps: [B,N,W,H,3] in 0-1 range - if None the texture will be metallic
            - cameras: PerspectiveCamera object
            - color: [B,N,V,3]
        Return:
            - img: [B,W,H,C]
        """

        if isinstance(vertices, torch.Tensor):
            _, N, V, _ = vertices.size()
            list_faces = []
            list_vertices = []
            for i in range(N):
                list_faces.append(faces + V * i)
                list_vertices.append(vertices[:, i])
            faces = torch.cat(list_faces, 1)  # [B,N*F,3]
            vertices = torch.cat(list_vertices, 1)  # [B,N*V,3]

            # Metallic texture
            verts_rgb = torch.ones_like(vertices).reshape(-1, N, V, 3)  # [1,N,V,3]
            if color is not None:
                verts_rgb = color * verts_rgb
            verts_rgb = verts_rgb.flatten(1, 2)
            textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb)
            
            # Create meshes
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
        else:
            # UV MAP
            if color is not None and len(color[0].shape) == 3:
                textures = pytorch3d.renderer.TexturesUV(maps=color, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
            else:
                tex = [torch.ones_like(vertices[i]) if color is None else torch.ones_like(vertices[i]) * color[i] for i in range(len(vertices))]
                tex = torch.cat(tex)[None]
                textures = pytorch3d.renderer.Textures(verts_rgb=tex)
            
            verts = torch.cat(vertices)

            faces_up = []
            n = 0
            for i in range(len(faces)):
                faces_i = faces[i] + n
                faces_up.append(faces_i)
                n += vertices[i].shape[0]
            faces = torch.cat(faces_up)
            meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=textures)
        
        return self.add_light_and_render(meshes, cameras)

    def add_light_and_render(self, meshes, cameras):
        lights = PointLights(device=meshes.device, ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[0, 0, 0]], specular_color=[[0, 0, 0]], location=((0, 0, 0), ))
        images, frags = self.renderer(meshes, cameras=cameras, lights=lights)

        rgb_images = images[..., :3]
        rgb_images = torch.clamp(rgb_images, 0., 1.)
        rgb_images = rgb_images * 255
        rgb_images = rgb_images.to(torch.uint8)
            
        return rgb_images, frags.zbuf[0]

    def renderPerspective(self, vertices, faces, camera_translation, principal_point=None, color=None, rotation=None,
                          focal_length=2 * 500. / 500., # 2 * focal_length / image_size
                          K=None,
                          faces_uvs=None,
                          verts_uvs=None,
                          render_fn='render',
                          # with cameras
                          textures=None,
                          ):
        """
        Args:
            - vertices: [B,V,3] or [B,N,V,3] where N is the number of persons OR list of tensor of shape [V,3]
            - faces: [B,13776,3] OR list of tensor of shape [V,3]
            - focal_length: float
            - principal_point: [B,2]
            - T: [B,3]
            - color: [B,N,3]
        Return:
            - img: [B,W,H,C] in range 0-1
        """

        device = vertices[0].device

        if principal_point is None:
            principal_point = torch.zeros_like(camera_translation[:, :2])

        if isinstance(vertices, torch.Tensor) and vertices.dim() == 3:
            vertices = vertices.unsqueeze(1)

        # Create cameras
        if rotation is None:
            R = self.base_rotation
        else:
            R = torch.bmm(self.base_rotation, rotation)
        camera_translation = torch.einsum('bik, bk -> bi', self.base_rotation.repeat(camera_translation.size(0), 1, 1),
                                          camera_translation)
        if self.convention == 'opencv':
            principal_point = -torch.as_tensor(principal_point)
        
        cameras = pytorch3d.renderer.PerspectiveCameras(
                                                        R=R, T=camera_translation, device=device,
                                                        K=K,
                                                        focal_length=focal_length, principal_point=principal_point,
                                                        )

        if render_fn == 'render':
            rgb_images, depth_images = self.render(vertices, faces, cameras, color, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        else:
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
            return self.add_light_and_render(meshes, cameras)

        return rgb_images, depth_images

def check_scale(verts):
    if (verts.max() - verts.min()) > 1:
        print('rescale')
        verts = verts / 1000.
    return verts

@torch.no_grad()
def render_comb_mesh(comb_mesh, save_base_dir, sqn_dir, image_size = 1280, t_start=0, t_end=1000000):
    verts = comb_mesh.verts_list()[0]
    faces = comb_mesh.faces_list()[0]
    
    verts = check_scale(verts)

    os.makedirs(save_base_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    renderer = PyTorch3DRenderer(image_size=image_size,
                 blur_radius=0.00001,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                 background_color=(0, 0, 0),
                #  background_color=(255, 255, 255),
    ).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    pose_dir = osp.join(sqn_dir, 'icp_res')
    rgb_dir = osp.join(sqn_dir, 'rgb')
    rgb_imgs = os.listdir(rgb_dir)
    rgb_imgs.sort()
    subdirs = os.listdir(pose_dir)
    subdirs.sort()
    
    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
    )

    resln = torch.Tensor([1280, 720]).float()
    ratio_render = resln[0] / image_size
    delta = int(((resln[0] - resln[1]) // 2) / ratio_render)
    principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / resln - 0.5) * 2).reshape(1,2)
    focal_length = torch.Tensor([[(2 * intrinsics[0,0] / resln[0]), (2 * intrinsics[1,1] / resln[0])]]) # TODO check documentation
    subdirs = subdirs[t_start:t_end]

    for t, subdir in enumerate(tqdm(subdirs)):
        # update the mesh according to the homogenous matrix
        fname = os.path.join(pose_dir, subdir, 'f_trans.txt')
        mat = torch.from_numpy(np.loadtxt(fname)).float()
        mat = torch.inverse(mat)
        verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
        verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
        verts_up = verts_up[:,:3,0]
        # rendering the mesh - no texture
        texture = comb_mesh.textures.verts_features_list()[0]
        img_mesh, dimg_mesh = renderer.renderPerspective(vertices=[verts_up.to(device)],
                                            faces=[faces.to(device)],
                                            color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            # faces_uvs=[faces_uvs.to(device)],
                                            # verts_uvs=[verts_uvs.to(device)],
                                            )
        
        img_mesh = img_mesh.cpu().numpy()[0][delta : -delta]
        dimg_mesh = dimg_mesh.cpu().numpy()[delta : -delta]
        dimg_mesh = (dimg_mesh * 255).astype(np.uint8)

        fn_img = os.path.join(save_base_dir, 'images', f"{t:010d}.png")
        os.makedirs(osp.dirname(fn_img), exist_ok=True)
        Image.fromarray(img_mesh).save(fn_img)


@torch.no_grad()
def render_mhand_obj_frames(sqn=None, datadir=None, savedir=None):
    sqn_dir = osp.join(datadir, sqn)

    obj_mesh_pth = osp.join(datadir, sqn, 'gt_mesh/obj_mesh/objmesh.obj')
    hand_mesh_pth = osp.join(datadir, sqn, 'gt_mesh/mano_mesh/mano.obj')

    renders_save_dir = osp.join(savedir, sqn, 'ho')
    
    obj_vertices_, obj_faces_, obj_texcoords_, obj_texInd_, obj_texture_ = load_mesh(osp.dirname(obj_mesh_pth), osp.basename(obj_mesh_pth))
    obj_faces = torch.from_numpy(np.array(obj_faces_, dtype=np.int32))
    
    reg_res = json.load(open(osp.join(datadir, sqn, 'registration.json')))
    obj_pose = np.array(reg_res['object_pose'])
    new_verts = trnsfm_points(obj_pose, obj_vertices_)
    # new_verts = check_scale(new_verts)
    obj_verts = torch.from_numpy(new_verts).float()
    obj_texture = pytorch3d.renderer.TexturesVertex(verts_features=torch.repeat_interleave(torch.tensor([1, 0, 0]).unsqueeze(0), obj_vertices_.shape[0], 0).unsqueeze(0))
    obj_mesh = pytorch3d.structures.Meshes(verts=[obj_verts], faces=[obj_faces], textures=obj_texture)
    
    hand_vertices_, hand_faces_, hand_texcoords_, hand_texInd_, hand_texture_ = load_mesh(osp.dirname(hand_mesh_pth), osp.basename(hand_mesh_pth))
    hand_faces = torch.from_numpy(np.array(hand_faces_, dtype=np.int32))
    # hand_vertices_ = check_scale(hand_vertices_)
    hand_verts = torch.from_numpy(hand_vertices_).float()
    hand_texture = pytorch3d.renderer.TexturesVertex(verts_features=torch.repeat_interleave(torch.tensor([0, 0, 1]).unsqueeze(0), hand_vertices_.shape[0], 0).unsqueeze(0))
    hand_mesh = pytorch3d.structures.Meshes(verts=[hand_verts], faces=[hand_faces], textures=hand_texture)
    # bb()
    comb_mesh = pytorch3d.structures.join_meshes_as_scene([hand_mesh, obj_mesh])
    render_comb_mesh(comb_mesh, renders_save_dir, sqn_dir, image_size = 1280, t_start=0, t_end=1000000)

    print('Done!')    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rendering')
    parser.add_argument('--datadir', type=str, default='/scratch/2/data/SHOWMe', help='dataset root dir pth') 
    parser.add_argument('--outdir', type=str, default='./out', help='output') 
    parser.add_argument('--seq_id', type=str, default=None, help='sequence id')
    args = parser.parse_args()

    render_mhand_obj_frames(sqn=args.seq_id, datadir=args.datadir, savedir=args.outdir)