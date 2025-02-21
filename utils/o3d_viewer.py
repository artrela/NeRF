from dataset.dataloader import SyntheticDataloader
import open3d as o3d
import numpy as np
import random
from typing import List
import viewer_utils
import os
from PIL import Image
import tqdm
import argparse
import render_utils
from scipy.spatial.transform import Rotation as R

def create_cameras(poses: List[np.ndarray], K: np.ndarray, img_size: int=800)->List[o3d.geometry.LineSet]:
    """ Create camera visualizations using o3d interface

    Args:
        poses (List[np.ndarray]): A list of k sample images from the dataset
        K (np.ndarray): The camera's intrinstic matrix
        img_size (int, optional): image size in the o3d camera line set. Defaults to 800.

    Returns:
        List[o3d.geometry.LineSet]: A list of camera lineset objects
    """
    camera_visuals = []
    
    for world2cam in poses:
        
        cam2world = np.eye(4)
        world2cam_t = world2cam[:3, 3]
        world2cam_R = world2cam[:3, :3]
        
        # o3d faces cam along +z, this dataset does -z
        world2cam_R[:, -1] *= -1
        
        cam2world[:3, :3] =   world2cam_R.T
        cam2world[:3,  3] = - world2cam_R.T @ world2cam_t
        
        camera = o3d.geometry.LineSet.create_camera_visualization(img_size, img_size, K, cam2world)
        camera_visuals.append(camera)
    
    return camera_visuals

def create_images(images: List[np.ndarray], poses: List[np.ndarray])->List[o3d.geometry.TriangleMesh]:
    """ Create an image plane (front & back) to visualize on each camera interface. See 
    # https://stackoverflow.com/a/77331860 for more details

    Args:
        images (List[np.ndarray]): A list of k sample images from the dataset
        poses (List[np.ndarray]): A list of k sample poses that match the images from the dataset

    Returns:
        List[o3d.geometry.TriangleMesh]: Image planes for each image, camera pair
    """
    
    images_visuals = []
    
    img_creator = viewer_utils.CreateO3DImage()
    for image, world2cam in zip(images, poses):
        
        img_obj = img_creator.createGeometry(image)
        
        img_plane = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(img_creator.getVerts()), 
                o3d.utility.Vector3iVector(img_creator.getFaces()))
        img_plane.triangle_uvs = o3d.utility.Vector2dVector(img_creator.getUV())
        
        img_plane.textures = [img_obj]
        img_plane.triangle_material_ids = o3d.utility.IntVector([0]*len(img_creator.getFaces()))
        
        img_plane.translate(-img_plane.get_center())
        img_plane.transform(world2cam)
        # recall +z in our coordinates is -z in o3d
        img_plane.translate((world2cam @ [0, 0, -1, 1])[:3], relative=False)
    
        images_visuals.append(img_plane)
        
    return images_visuals

def create_hotdog()->List[o3d.geometry.TriangleMesh]:
    '''
    Create a hotdog from the original nerf blender file mentioned in their README.md
    
    Required manual color assignment to vertices as mentioned in this git issue.
    https://github.com/isl-org/Open3D/issues/2688#issuecomment-748679275
    '''
    # 
    hot_dog = o3d.io.read_triangle_mesh("data/nerf_synthetic/hotdog/hot-dogs.obj")

    colors = [
        [0.787967, 0.322304, 0.079918], # "Chipotle":     
        [0.95, 0.95, 0.95], # Plate   
        [0.95, 0.95, 0.95], # Plate 
        [1.0, 0.747158, 0.1972], # "Mayo":     
        [0.799943, 0.460018, 0.06398], # Bread:         
        [0.799943, 0.460018, 0.06398], # Bread:         
        [0.799103, 0.088655, 0.0185], # Hot Dog:    
        [0.266787, 0.018357, 0.003465] # "Salsa":        
    ]

    # The obj contained materials, with no vertex color assigned
    # iterate through materials and assign colors to the vertices
    vertex_colors = np.zeros((len(hot_dog.vertices), 3))
    for id in np.unique(hot_dog.triangle_material_ids):
        color = colors[id-1]
        for tri in np.where(hot_dog.triangle_material_ids == id)[0]:
            for vert in hot_dog.triangles[tri]:
                vertex_colors[vert] = color
        
    hot_dog.compute_vertex_normals()
    hot_dog.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    hot_dog.scale(10, hot_dog.get_center()) # arbitrary scaling to fit scene
    
    return [hot_dog]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--headless", action="store_true",
                        help="Don't launch visualizer, generate an animation.")
    parser.add_argument("-v", "--views", type=int, default=10, required=False,
                        help="The number of camera poses to visualize.")
    parser.add_argument("-r", "--rays", type=int, default=0, required=False,
                        help="The number of rays to visualize, per camera.")
    args = parser.parse_args()
    
    print ('\n =============================== ')
    print (f' ===  Local Open3D=={o3d.__version__} === ')
    print (f' ===  Tested Open3D==0.19.0 === ')
    print (' =============================== \n')
    
    dataset = SyntheticDataloader(pth="data/nerf_synthetic/", item="hotdog", split="train")

    # unpack dataset images, poses
    poses, images, cam2img = dataset.transforms, dataset.images, dataset.cam2img
    
    samples = random.sample(list(range(len(dataset))), k=args.views)
    
    cameras = create_cameras(poses[samples], cam2img)
    images = create_images(images[samples], poses[samples])
    hotdog = create_hotdog()
    
    def direction_to_euler(d):
        r = R.from_rotvec(np.cross([0, 0, 1], d) * np.arccos(np.clip(d[2], -1.0, 1.0)))
        return r.as_matrix()
    
    def rot_pos_z():
        R = np.eye(3)
        R[2, 2] = -1
        return R
    
    def create_rays(dataset, poses, num_rays):
        
        if not num_rays:
            return []
        
        rays = []
        ray_pts = []
        for pose in poses:
            
            us, vs = np.random.randint(0, dataset.img_shape[0], size=(2, num_rays))
            
            ray_pts.append(us, vs)
            pose[:3, 2] *= -1 # cameras +z points along -z, align o3d coordinate frame
            # o, d = dataset.get_rays_np(pose)
            o, d = dataset.create_rays(pose[:3, :3], pose[:3, 3])
            o, d = o.cpu().numpy(), d.cpu().numpy()
            
            for idx, (u,v) in enumerate(zip(us, vs)):
                ray = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius = 0.01, 
                    cylinder_height = np.random.uniform(2, 6), 
                    cone_radius = 0.05, 
                    cone_height = 0.05, 
                    resolution = 20, 
                    cylinder_split = 4, 
                    cone_split = 1
                )
                ray.rotate(rot_pos_z(), (0,0,0)) # with our convention as camera facing along -z, 
                # start ray along -z axis
                ray.rotate(direction_to_euler(d[u,v]), (0,0,0))
                ray.translate(o[u,v])
                ray.paint_uniform_color(np.random.rand(3))
                # ray.transform(pose)
                rays.append(ray)
        
        return rays
        
    #TODO place points in the scene to see if that works well
    rays, ray_pxs = create_rays(dataset, poses[samples], args.rays)
    cf = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]
    
    def create_points(dataset, poses, uvs, Nc):
        
        pcd_pts = []
        
        for pose, (u, v) in zip(poses, uvs):
            pose[:3, 2] *= -1
            o, d = dataset.create_rays(pose[:3, :3], pose[:3, 3]) # (H, W, 3), (H, W, 3)
            
            t = render_utils.stratified_sampling_rays(tn=2., tf=6., N=Nc, rays=args.rays) # (rays, Nc samples along ray)

            x = o[u, None, v] + t[..., None]*d[u, None, v] # (rays, Nc samples, xyz)
            pcd_pts.append(x)
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcd_pts, 0))
        pcd.paint_uniform_color([0.5, 0.5, 0])        
        
        return [pcd]
    
    ray_pcd = create_points(dataset, poses[samples], ray_pxs, Nc=64)
    
    if args.headless:
        viewer_utils.create_animation(*hotdog, *cameras, *images, *rays, *ray_pcd)
    else:
        o3d.visualization.draw_geometries( hotdog + cameras + images + rays + ray_pcd + cf)
    