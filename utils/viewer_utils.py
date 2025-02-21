from PIL import Image
import numpy as np
import open3d as o3d
import time
import tqdm

def create_animation(*args):
    '''
    Create an animation rotating around the scene
    '''
    vis = o3d.visualization.Visualizer()  
    vis.create_window(visible=False, width=640, height=480)  
    ctr = vis.get_view_control()
    
    for obj in args:
        vis.add_geometry(obj)
    
    # https://github.com/isl-org/Open3D/issues/2139 - set axis then rotate
    ctr.set_up((0, 0, 1))
    ctr.set_front((0, 1, 0))
    ctr.set_zoom(0.50)
    
    imgs = []
    for _ in tqdm.tqdm(range(0, 300), desc="Rendering Scene"):
        ctr.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        img = (np.array(vis.capture_screen_float_buffer()) * 255.).astype('uint8')
        imgs.append(Image.fromarray(img))
    
    print("Saving gif...")
    start = time.time()
    imgs[0].save('example.gif', save_all = True, append_images = imgs[1:], optimize = True, duration = 10)
    print(f"Saved gif in: {time.time()-start:0.2f} seconds.")
    
    return 
    
class CreateO3DImage:
    
    '''
    Connect the vertices of the image plane. First two are front face of mesh, 
    second two have reverse winding order for connecting the vertices
    '''
    faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
        ])
    def getFaces(self):
        return self.faces
    
    '''
    The uv coordinates tell you where the texture will be mapped to. 
    For example, the first face will start sampling the face texture image
    in the top right hand corner, [0, 1]. 
    '''
    v_uv = np.array([   [0, 1], [1, 1], [1, 0], 
                        [0, 1], [1, 0], [0, 0],
                        [1, 1], [0, 0], [0, 1],
                        [1, 1], [1, 0], [0, 0]])
    def getUV(self):
        return self.v_uv
    
    '''
    The first four verts are the front plane and the second are the back plane, showing the 
    reverse side of the image. Small displacement to avoid aliasing effects.
    '''
    vertices = np.array([
            [-0.5, -0.5,     0],
            [ 0.5, -0.5,     0],
            [ 0.5,  0.5,     0],
            [-0.5,  0.5,     0],
            [-0.5, -0.5, -0.01],
            [ 0.5, -0.5, -0.01],
            [ 0.5,  0.5, -0.01],
            [-0.5,  0.5, -0.01],
        ])
    def getVerts(self):
        return self.vertices
    
    @staticmethod
    def createGeometry(img):
        return o3d.geometry.Image((img * 255.).astype(np.uint8))
    
    