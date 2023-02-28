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

# MODEL_PATH = "/home/michel_ma/MA_Heinemann/catkin_ws/src/rgbd_weight_estimation/resnext101_paper_final_best.pth"
MODEL_PATH = "/home/michel_ma/MA_Heinemann/catkin_ws/src/trained_models/final/resnext101_set1_final_best.pth"

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

# Verification process
def validate_model(net, device, test_loader, epoch):
    net.eval()
    acc = 0; acc_batch = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = net(data)
            print("output = %f   target = %f" % (output, target))
            acc_batch_sum = 0
            for i in range(len(output[:,0])):
                delta = np.absolute(output[i,0].cpu() - target[i,0].cpu())
                acc = 1.0 - (delta / target[i,0].cpu())
                acc_batch_sum += acc #np.clip(acc, 0, 1)
            
            acc_batch += (acc_batch_sum / BATCH_SIZE)
    acc = acc_batch / len(test_loader)
    print("epoch = %d   accuracy = %f" % (epoch, acc))
    return acc

# for i in range(2):
#     set_ = "set" + str(i+1)
#     for j in range(14):
#         if set_ == "set1" and j>7:
#             continue
#         model_name = "resnext101_" + set_ + "_final_eps" + str(j*10+10)
#         path_ = MODEL_PATH + model_name + ".pth"

#         print(str(j*10+10))
#         net = ResNeXt101(path= path_, grads=False)
#         net.to(DEVICE)
#         acc = validate_model(net, DEVICE, test_loader, 0)

#         now = datetime.now()
#         current_time = now.strftime("%H:%M:%S")

#         with open(('validation_4.txt'), 'a') as f:
#                 f.write(current_time + ": model: %s accuracy: %f \n" % (model_name, acc))


net = ResNeXt101(path= MODEL_PATH, grads=False)
net.to(DEVICE)
acc = validate_model(net, DEVICE, test_loader, 0)









