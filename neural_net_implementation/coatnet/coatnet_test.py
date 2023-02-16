import torch
from coatnet import coatnet_3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights

# path = "/media/michel_ma/NVMe/MA_Heinemann_Dataset/YCB-captures/0007_Banana/Banana000011.png"
path = "/home/michel/catkin_ws/src/MA_Heinemann/neural_net_implementation/dataset/test/Adjustable_Wrench000001.png"

img = Image.open(path).convert('RGB')

# transform = transforms.Compose([transforms.Resize([224, 224]),
#                                 transforms.ToTensor()])

weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2

transform = weights.transforms()

img = transform(img)

img = img.unsqueeze(0)

# net = coatnet_3()

net = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

# net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
net.eval()


prediction = net(img).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
