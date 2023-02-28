import os, json
from torch.utils import data
from sklearn.model_selection import train_test_split
import fnmatch
import torch
import cv2 as cv
import open3d as o3d
import numpy as np


# IMAGE_PATH_ROOT = "/media/michel_ma/NVMe2/Paper_Dataset/001_Train"
IMAGE_PATH_ROOT = "/media/michel_ma/NVMe2/Paper_Dataset/002_Eval"
# IMAGE_PATH_ROOT = "/media/michel_ma/NVMe2/Paper_Dataset/002_Eval/003_Pear/030_Pear_3"


SCALING = True

SCALING_RANGE = [0.5, 2.0]


### Realsense D435
DEPTH_FX = 378.704
DEPTH_FY = 378.704
DEPTH_CX = 317.212 # 320
DEPTH_CY = 235.901 # 240

DEPTH_IN_PARAMS = np.array([[DEPTH_FX, 0, DEPTH_CX], [0, DEPTH_FY, DEPTH_CY], [0, 0, 1]])

class SeedlingData (data.Dataset):

    def __init__(self, root=IMAGE_PATH_ROOT, train=True, eval=False, split=True, data_augment=True):

        paths = list()
        for path, _, files in os.walk(root):
            paths += [os.path.join(path, f[:-4]) for f in files if fnmatch.fnmatch(f, '*[0-9][0-9][0-9].png')]

        self.data_augment = data_augment

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
            labels = [float(data[os.path.basename(f)[:-7]]) for f in files]

        self.labels = labels

        # for color_path in self.color_images:
        #     roi_path = os.path.dirname(color_path) + '/roi.json'
        #     print(roi_path)
        #     with open(roi_path, 'r') as json_file:
        #         filedata = json_file.read()

        #     # Replace the target string
        #     filedata = filedata.replace('}{', ',')

        #     with open(roi_path, 'w') as file:
        #         file.write(filedata)

        print("correct!")


    def __getitem__(self, index):

        if self.data_augment:
            scaling_factor = np.min(SCALING_RANGE) + ((np.max(SCALING_RANGE) - np.min(SCALING_RANGE)) * np.array(np.random.rand()).round(3))
        else:
            scaling_factor = 1.0

        label = torch.as_tensor([self.labels[index]]) * scaling_factor

        def processing(idx, scaling_factor, data_augment):
            depth_path = self.depth_images[idx]
            color_path = self.color_images[idx]
            depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH)

            color = cv.imread(color_path, cv.IMREAD_COLOR)

            pc = self.create_point_cloud(DEPTH_IN_PARAMS, depth[:,:].astype(np.uint16)) * scaling_factor

            # remove distance to camera
            pc[:,:,2] -= np.mean(pc[:,:,2])

            if data_augment:
                roi_path = os.path.dirname(color_path) + '/roi.json'

                with open(roi_path) as json_file:
                    data = json.load(json_file)
                    roi = np.asarray(data[os.path.basename(color_path)[:-4]])

                x1 = np.min([roi[0], roi[2]])
                y1 = np.min([roi[1], roi[3]])
                x2 = np.max([roi[0], roi[2]])
                y2 = np.max([roi[1], roi[3]])

                left = np.random.randint(0, int(x1))
                right = np.random.randint(int(x2), pc.shape[1])
                top = np.random.randint(0, int(y1))
                bottom = np.random.randint(int(y2), pc.shape[0])

                pc = pc[left:right, top:bottom, :]
                color = color[left:right, top:bottom, :]

            depth = cv.resize(pc, [224, 224])
            color = cv.resize(color, [224, 224])

            depth = np.reshape(depth, [3, 224, 224])     
            color = np.reshape(color, [3, 224, 224])

            depth = torch.tensor(depth)
            color = torch.tensor(color)

            return depth, color
            
        try:
            depth, color = processing(index, scaling_factor, self.data_augment)

        except:
            # in case image may be corrupted, just use the image before again to proceed training
            print(self.depth_images[index])
            print(self.color_images[index])
            depth, color = processing(index-1, scaling_factor, self.data_augment)

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
    # rnd_idx = 5996
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

    
