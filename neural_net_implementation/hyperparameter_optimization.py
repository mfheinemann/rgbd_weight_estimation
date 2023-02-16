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
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights, resnext101_64x4d, ResNeXt101_64X4D_Weights

from coatnet.coatnet import CoAtNet

import numpy as np

from copy import deepcopy

# Set global parameters
modellr = [1e-4, 1e-2]
loss_func = nn.MSELoss()
BATCH_SIZE = [16, 32]
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_rgb = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor()])
                                    
transform_depth = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor()])

# Read data
dataset_train = SeedlingData(train=True, eval=False)
dataset_test = SeedlingData(train=False, eval=True)


### BEGIN NET MODIFICATION ###
def resnext101_64x4():
    net = resnext101_64x4d()

    # change first layer to take 6 dim image
    weight = net.conv1.weight.clone()
    net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # change last layer to single output neuron
    lin = net.fc
    new_lin = nn.Sequential(
        lin,
        nn.Linear(lin.out_features, 1, bias=True)
    )
    net.fc = new_lin

    with torch.no_grad():
        net.conv1.weight[:, :3] = weight
        net.conv1.weight[:, 3:] = net.conv1.weight[:, :3]

    return net

def resnext101_32x8():
    net = resnext101_32x8d()

    # change first layer to take 6 dim image
    weight = net.conv1.weight.clone()
    net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # change last layer to single output neuron
    lin = net.fc
    new_lin = nn.Sequential(
        lin,
        nn.Linear(lin.out_features, 1, bias=True)
    )
    net.fc = new_lin

    with torch.no_grad():
        net.conv1.weight[:, :3] = weight
        net.conv1.weight[:, 3:] = net.conv1.weight[:, :3]

    return net

def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    net = CoAtNet((224, 224), 6, num_blocks, channels, num_classes=768)
    lin = net.fc
    new_lin = nn.Linear(lin.out_features, 1, bias=True)
    net.fc = new_lin
    return net

def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    net = CoAtNet((224, 224), 6, num_blocks, channels, num_classes=768)
    lin = net.fc
    new_lin = nn.Linear(lin.out_features, 1, bias=True)
    net.fc = new_lin
    return net

def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    net = CoAtNet((224, 224), 6, num_blocks, channels, num_classes=1026)
    lin = net.fc
    new_lin = nn.Linear(lin.out_features, 1, bias=True)
    net.fc = new_lin
    return net


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    net = CoAtNet((224, 224), 6, num_blocks, channels, num_classes=1536)
    lin = net.fc
    new_lin = nn.Linear(lin.out_features, 1, bias=True)
    net.fc = new_lin
    return net

def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    net = CoAtNet((224, 224), 6, num_blocks, channels, num_classes=1536)
    lin = net.fc
    new_lin = nn.Linear(lin.out_features, 1, bias=True)
    net.fc = new_lin
    return net


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
def validate_model(net, device, test_loader, epoch, batch_size):
    net.eval()
    acc = 0; acc_batch = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = net(data)
            acc_batch_sum = 0
            # abs_delta = np.absolute(output.sum().cpu() - target.sum().cpu())
            # acc_batch += 1.0 - (abs_delta / target.sum().cpu())
            for i in range(len(output[:,0])):
                delta = np.absolute(output[i,0].cpu() - target[i,0].cpu())
                acc = 1.0 - (delta / target[i,0].cpu())
                acc_batch_sum += acc
            
            acc_batch += (acc_batch_sum / batch_size)
    acc = acc_batch / len(test_loader)
    print("epoch = %d   accuracy = %f" % (epoch, acc))
    return acc


nets = [resnext101_32x8(), resnext101_64x4(),coatnet_0(), coatnet_1(), coatnet_2(), coatnet_3(), coatnet_4()]
model_names = ['resnext101_32x8', 'resnext101_64x4','coatnet_0', 'coatnet_1', 'coatnet_2', 'coatnet_3', 'coatnet_4']
training_sequences = ['P112', 'P122', 'P212', 'P222']

i = 0

for net_org in nets:

    j = 0

    for batch_size in BATCH_SIZE:

        for lr in modellr:

            # Import data
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

            net = deepcopy(net_org)

            net.to(DEVICE)

            optimizer = optim.Adam(net.parameters(), lr=lr)

            # train
            best_acc = 0
            model_name = model_names[i]
            training_sequence = training_sequences[j]

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            with open((model_name + '_' + training_sequence + '_training_log.txt'), 'a') as f:
                f.write(current_time + ": model: %s sequence: %s lr: %2f batch_size: %2d \n" % (model_name, training_sequence, lr, batch_size))

            for epoch in range(1, EPOCHS + 1):
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time + ": training epoch %2d" % (epoch))
                loss = train_one_epoch(net, DEVICE, train_loader, optimizer, epoch)
                acc = validate_model(net, DEVICE, test_loader, epoch, batch_size)
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

            del net
            del train_loader
            del test_loader
            
            j += 1

    i += 1


