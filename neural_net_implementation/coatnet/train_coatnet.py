from datetime import datetime
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset import SeedlingData
# from dataset.dataset import SeedlingData
from torch.autograd import Variable
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights

from coatnet.coatnet import CoAtNet

import numpy as np

# Set global parameters
modellr = 1e-4
loss_func = nn.MSELoss()
BATCH_SIZE = 16 
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_rgb = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor()])
                                    
transform_depth = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor()])

# Read data
dataset_train = SeedlingData(transforms_rgb=transform_rgb, transforms_depth=transform_depth, train=True, eval=False)
dataset_test = SeedlingData(transforms_rgb=transform_rgb, transforms_depth=transform_depth, train=False, eval=True)
# Import data
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


### BEGIN NET MODIFICATION ###



def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 6, num_blocks, channels, num_classes=768)

def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 6, num_blocks, channels, num_classes=1026)

def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 6, num_blocks, channels, num_classes=1536)

def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 6, num_blocks, channels, num_classes=1536)


# Define training process

def train_one_epoch(net, device, train_loader, optimizer, epoch):
    net.train()
    epoch_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = net(data)

        loss = loss_func(output, target)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        if (batch_idx) % 10 == 0:
            print("epoch = %4d   batch = %4d   batch_loss = %0.4f" % (epoch, batch_idx, loss.item()))

    print("epoch = %4d   epoch_loss = %0.4f" % (epoch, epoch_loss/len(train_loader)))

    return epoch_loss/len(train_loader)

# Verification process
def validate_model(net, device, test_loader, epoch, pst):
    net.eval()
    acc = 0; acc_batch = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = net(data)            
            abs_delta = np.absolute(output.sum().cpu() - target.sum().cpu())
            acc_batch += 1.0 - (abs_delta / target.sum().cpu())
    acc = acc_batch / len(test_loader)
    print("epoch = %d   accuracy = %f" % (epoch, acc))
    return acc

# change last layer to single output neuron
# nets = [coatnet_1(), coatnet_2(), coatnet_3(), coatnet_4()]
# model_names = ['coatnet_1', 'coatnet_2', 'coatnet_3', 'coatnet_4']

nets = [coatnet_2()]
model_names = ['coatnet_2']

i = 0

for net in nets:
    lin = net.fc
    new_lin = nn.Linear(lin.out_features, 1, bias=True)
    net.fc = new_lin

    net.to(DEVICE)
    ### END NET MODIFICATION ###

    optimizer = optim.Adam(net.parameters(), lr=modellr)

    # train
    best_acc = 0
    model_name = model_names[i] + "_set1"
    training_sequence = "t1"
    for epoch in range(1, EPOCHS + 1):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time + ": training epoch %2d" % (epoch))
        loss = train_one_epoch(net, DEVICE, train_loader, optimizer, epoch)
        acc = validate_model(net, DEVICE, test_loader, epoch, 0.10)
        with open((model_name + '_' + training_sequence + '_training_log.txt'), 'a') as f:
            f.write(current_time + ": epoch: %2d loss: %2f acc: %2f \n" % (epoch, loss, acc))
        if acc >= best_acc:
            print("new acc: %2f better than: %2f  -> saving model at epoch: %4d" % (acc, best_acc, epoch))
            torch.save(net.state_dict(), (model_name + '_' + training_sequence + '_best.pth'))
            ### logging
            with open((model_name + '_' + training_sequence + '_training_log.txt'), 'a') as f:
                f.write(current_time + ": new acc: %2f better than: %2f  -> saving model at epoch: %4d\n" % (acc, best_acc, epoch))
            best_acc = acc
        if epoch % 10 == 0:
            print("current acc: %2f -> saving model at after epoch: %4d" % (acc, epoch))
            torch.save(net.state_dict(), (model_name + '_' + training_sequence + '_eps' + str(epoch) + '.pth'))
            with open((model_name + '_' + training_sequence + '_training_log.txt'), 'a') as f:
                f.write(current_time + ": current acc: %2f -> saving model at after epoch: %4d \n" % (acc, epoch))

    i += 1
