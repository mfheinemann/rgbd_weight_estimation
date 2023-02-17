import os, json
from torch.utils import data
from sklearn.model_selection import train_test_split
import fnmatch
import torch
import cv2 as cv
import open3d as o3d
import numpy as np


IMAGE_PATH_ROOT = "/media/michel_ma/NVMe2/MA_Heinemann_Dataset/000_All_data/001_Real"

SCALING = True

SCALING_RANGE = [0.5, 2.0]


### Realsense D435
DEPTH_FX = 378.704
DEPTH_FY = 378.704
DEPTH_CX = 317.212 # 320
DEPTH_CY = 235.901 # 240

DEPTH_IN_PARAMS = np.array([[DEPTH_FX, 0, DEPTH_CX], [0, DEPTH_FY, DEPTH_CY], [0, 0, 1]])

class SeedlingData (data.Dataset):

    def __init__(self, root=IMAGE_PATH_ROOT, train=True, eval=False, split=True):

        paths = list()
        for path, _, files in os.walk(root):
            paths += [os.path.join(path, f[:-4]) for f in files if fnmatch.fnmatch(f, '*[0-9][0-9][0-9].png')]

        if split:
            # for training
            trainval_names, val_names = train_test_split(paths, test_size=0.4, random_state=42)
            eval_names, test_names = train_test_split(val_names, test_size=0.5, random_state=42)

            if train:
                files = trainval_names
            elif eval:
                files = eval_names
            else:
                files = test_names
        else:
            files = paths
        
        self.depth_images = [f + '.depth.png' for f in files]
        self.color_images = [f + '.png' for f in files]

        label_path = os.path.join(IMAGE_PATH_ROOT, 'labels.json')

        labels = list()
        with open(label_path) as json_file:
            data = json.load(json_file)
            labels = [float(data[os.path.basename(f)[:-6]]) for f in files]

        self.labels = labels

    def __getitem__(self, index):

        scaling_factor = np.min(SCALING_RANGE) + ((np.max(SCALING_RANGE) - np.min(SCALING_RANGE)) * np.array(np.random.rand()).round(3))

        label = torch.as_tensor([self.labels[index]]) * scaling_factor

        def processing(idx, scaling_factor):
            depth_path = self.depth_images[idx]
            color_path = self.color_images[idx]
            depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH)

            color = cv.imread(color_path, cv.IMREAD_COLOR)

            pc = self.create_point_cloud(DEPTH_IN_PARAMS, depth[:,:].astype(np.uint16)) * scaling_factor

            # remove distance to camera
            pc[:,:,2] -= np.mean(pc[:,:,2])

            depth = cv.resize(pc, [224, 224])     
            color = cv.resize(color, [224, 224])

            depth = np.reshape(depth, [3, 224, 224])     
            color = np.reshape(color, [3, 224, 224])

            depth = torch.tensor(depth)
            color = torch.tensor(color)

            return depth, color
            
        try:
            depth, color = processing(index, scaling_factor)

        except:
            # in case image may be corrupted, just use the image before again to proceed training
            print(self.depth_images[index])
            print(self.color_images[index])
            depth, color = processing(index-1)

        data = torch.empty(color.size(dim=0)+depth.size(dim=0), color.size(dim=1), color.size(dim=2))
        data[:3, :, :] = color
        data[3:, :, :] = depth

        return data, label

    def __len__(self):
        return len(self.labels)

    def create_point_cloud(self, in_params, depth_image):
        image_dim = depth_image.shape

        intr = o3d.camera.PinholeCameraIntrinsic()
        intr.set_intrinsics(image_dim[0], image_dim[1], in_params[0,0], in_params[1,1], in_params[0,2], in_params[1,2])

        # PC form depth image
        pcl = o3d.geometry.PointCloud()
        pcl = pcl.create_from_depth_image(o3d.geometry.Image(depth_image), intr, project_valid_depth_only = False)

        # flip the orientation, so it looks upright, not upside-down
        pcl_points = np.asanyarray(pcl.points)
        point_cloud_array = np.int16(1000*pcl_points.reshape(image_dim[0], image_dim[1], 3))

        return point_cloud_array

if __name__ == "__main__":
    dataset_test = SeedlingData(train=False, eval=False)
    rnd_idx = np.random.randint(0, len(dataset_test)-1)
    rnd_idx = 5996
    print(rnd_idx)

    img, label = dataset_test[rnd_idx]

    pc_points = img[3:,:,:].reshape(224*224, 3)
    print(pc_points)

    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    color_image = np.array(img[:3, :, :], dtype=np.uint8).reshape(224,224,3)
    color_image[:,:,[2,0]] = color_image[:,:,[0,2]]
    ax.imshow(color_image)

    ax = fig.add_subplot(2, 2, 2)
    depth_image = np.array(pc_points[:,2].cpu().detach().numpy()).reshape(224,224)
    depth_image = ((depth_image/np.max(depth_image)) * 255).astype(np.uint8)
    ax.imshow(np.array(depth_image, dtype=np.uint8))

    ax = fig.add_subplot(2,2,3, projection='3d')
    ax.scatter(pc_points[:,0], pc_points[:,1], pc_points[:,2])

    plt.show()

    source_points = o3d.geometry.PointCloud()
    source_points.points = o3d.utility.Vector3dVector(pc_points)

    o3d.visualization.draw_geometries([source_points])

    
