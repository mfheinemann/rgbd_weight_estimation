import rospy
from torchvision.models import resnext101_32x8d
from datetime import datetime
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset import SeedlingData
from torch.autograd import Variable
import numpy as np
from neural_net_implementation.srv import EvalModel, EvalModelResponse
from sensor_msgs.msg import Image
import open3d as o3d
from cv_bridge import CvBridge
import cv2 as cv

MODEL_PATH = "/home/michel_ma/MA_Heinemann/catkin_ws/src/resnext101_set2_final_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.MSELoss()

DEPTH_FX = 378.704
DEPTH_FY = 378.704
DEPTH_CX = 317.212 # 320
DEPTH_CY = 235.901 # 240

CAMERA_INSET = 100

DEPTH_IN_PARAMS = np.array([[DEPTH_FX, 0, DEPTH_CX], [0, DEPTH_FY, DEPTH_CY], [0, 0, 1]])

class ResNeXt101(nn.Module):
    def __init__(self, path=MODEL_PATH, grads=True) -> None:
        super().__init__()
        
        self.grads = grads
        self.net = resnext101_32x8d()

        # change first layer to take 6 dim image
        self.net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # change last layer to single output neuron
        lin = self.net.fc
        new_lin = nn.Sequential(
            lin,
            nn.Linear(1000, 1, bias=True)
        )
        self.net.fc = new_lin

        self.net.load_state_dict(torch.load(path, map_location=DEVICE))

        self.gradients = None 

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        # register the hook
        if self.grads:
            x.register_hook(self.activations_hook)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        return x

class ResNeXtNode():

    def __init__(self) -> None:
        rospy.init_node("ResNeXtNode")

        self.model_path = MODEL_PATH
        self.device = DEVICE
        self.cv_bridge = CvBridge()
        self.data = torch.empty(6, 224, 224)
        
        self.net = ResNeXt101(grads=False)
        self.net.to(DEVICE)

        self.img_hist = torch.empty((3, 3, 224, 224))
        self.depth_hist = torch.empty((3, 3, 224, 224))

        rospy.Service('eval_model', EvalModel, self.handle_eval_model)

        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.handle_depth_image)
        rospy.Subscriber('/camera/color/image_raw', Image, self.handle_color_image)

        rospy.spin()

    def handle_eval_model(self, _):
        self.net.eval()
        self.data[:3, :, :] = torch.mean(self.img_hist, dim=0)
        self.data[3:, :, :] = torch.mean(self.depth_hist, dim=0)
        with torch.no_grad():
            data = Variable(self.data).to(self.device)
            output = self.net(data.unsqueeze(0))
            rospy.logwarn(output)

        return EvalModelResponse(float(output))

    def handle_depth_image(self, image):
        img = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        pc = self.create_point_cloud(DEPTH_IN_PARAMS, img.astype(np.uint16))
        pc = pc[CAMERA_INSET:-CAMERA_INSET, CAMERA_INSET:-CAMERA_INSET]
        img = self.reshaping(pc)
        self.depth_hist = torch.cat((self.depth_hist[1:], img.unsqueeze(0)), 0)
        
        #self.data[3:, :, :] = img

    def handle_color_image(self, image):
        img = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        img = img[CAMERA_INSET:-CAMERA_INSET, CAMERA_INSET:-CAMERA_INSET]
        img = self.reshaping(img)
        self.img_hist = torch.cat((self.img_hist[1:], img.unsqueeze(0)), 0)

        #self.data[:3, :, :] = img

    def reshaping(self, img_in):
        img = cv.resize(img_in, [224, 224])
        img = np.reshape(img, [3, 224, 224])
        img = torch.tensor(img)
        return img

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
    ResNeXtNode()