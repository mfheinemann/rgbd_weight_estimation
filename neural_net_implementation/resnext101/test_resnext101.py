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
from torchvision.models import resnext101_32x8d
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2 as cv

# Set global parameters
modellr = 1e-4
loss_func = nn.MSELoss()
BATCH_SIZE = 1
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "/home/michel_ma/MA_Heinemann/catkin_ws/src/rgbd_weight_estimation/resnext101_paper_final_best.pth"

# Read data
dataset_test = SeedlingData(train=False, eval=False, split=False) #eval=False

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

### BEGIN NET MODIFICATION ###
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

### END NET MODIFICATION ###

def test_model(net, device, dataset_test, epoch):
    net.eval()
    
    rnd_idx = np.random.randint(0, len(dataset_test)-1)
    # rnd_idx = 11
    print(rnd_idx)

    data, label = dataset_test[rnd_idx]
    img = deepcopy(data)

    data = Variable(data).to(device)
    data.requires_grad = True
    output = net(data.unsqueeze(0))    

    image = np.array(img[:3, :, :].cpu().detach().numpy(), dtype=np.uint8).reshape(224,224,3)

    pc_points = img[3:,:,:].reshape(224*224, 3)
    depth_image = np.array(pc_points[:,2].cpu().detach().numpy()).reshape(224,224)
    depth_image = ((depth_image/np.max(depth_image)) * 255).astype(np.uint8)

    print("label = %f   output = %f" % (label, output))

    output.backward()

    ### Saliency Maps
    grads = data.grad

    saliency_depth, _ = torch.max(grads[3:,:,:].abs(), dim=0) 
    saliency_depth = saliency_depth.reshape(224, 224)
    saliency, _ = torch.max(grads[:3,:,:].abs(), dim=0) 
    saliency = saliency.reshape(224, 224)

    saliency_img = cv.applyColorMap(cv.normalize(saliency.cpu().detach().numpy(), None, 255, 0, cv.NORM_MINMAX, cv.CV_8U), cv.COLORMAP_HOT)
    saliency_depth_img = cv.applyColorMap(cv.normalize(saliency_depth.cpu().detach().numpy(), None, 255, 0, cv.NORM_MINMAX, cv.CV_8U), cv.COLORMAP_HOT)

    ### Grad-CAM
    gradients = net.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = net.get_activations(data.unsqueeze(0)).detach()

    for i in range(2048):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)

    # normalize the heatmap
    heatmap /= np.max(heatmap)

    heatmap = cv.resize(heatmap, (img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    grad_cam_img = heatmap * 0.4 + image
    grad_cam_img = (((heatmap * 0.4 + image)/np.max(grad_cam_img)) * 255).astype(np.uint8)

    ### plots
    fig, ax = plt.subplots(2, 3)
    ax[0,0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax[0,0].axis('off')
    ax[0,1].imshow(cv.cvtColor(depth_image, cv.COLOR_GRAY2RGB))
    ax[0,1].axis('off')
    ax[1,0].imshow(cv.cvtColor(saliency_img, cv.COLOR_BGR2RGB))
    ax[1,0].axis('off')
    ax[1,1].imshow(cv.cvtColor(saliency_depth_img, cv.COLOR_BGR2RGB))
    ax[1,1].axis('off')
    ax[0,2].imshow(cv.cvtColor(grad_cam_img, cv.COLOR_BGR2RGB))
    ax[0,2].axis('off')

    text = "label: %f \n output: %f" % (np.round(label.cpu().detach().numpy(),2), np.round(output.cpu().detach().numpy(),2))
    ax[1,2].text(0.5, 0.5, text, horizontalalignment='center',
        verticalalignment='center')
    ax[1,2].axis('off')
    plt.tight_layout()
    plt.show()


# test
net = ResNeXt101(grads=True)
net.to(DEVICE)
test_model(net, DEVICE, dataset_test, 0)






