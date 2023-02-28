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
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
import numpy as np

# Set global parameters
modellr = 1e-4
loss_func = nn.MSELoss()
BATCH_SIZE = 32
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RETRAIN = True
MODEL_PATH = "/home/michel_ma/MA_Heinemann/catkin_ws/src/trained_models/final/resnext101_set1_final_best.pth"

# Read data
dataset_train = SeedlingData(train=True, eval=False)
dataset_test = SeedlingData(train=False, eval=True)
# Import data
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

### BEGIN NET MODIFICATION ###
if RETRAIN:
    net = resnext101_32x8d()
else:
    net = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

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

if RETRAIN:
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

else:
    with torch.no_grad():
        net.conv1.weight[:, :3] = weight
        net.conv1.weight[:, 3:] = net.conv1.weight[:, :3]

net.to(DEVICE)
### END NET MODIFICATION ###

optimizer = optim.Adam(net.parameters(), lr=modellr)

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
def validate_model(net, device, test_loader, epoch):
    net.eval()
    acc = 0; acc_batch = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = net(data)            
            acc_batch_sum = 0
            for i in range(len(output[:,0])):
                delta = np.absolute(output[i,0].cpu() - target[i,0].cpu())
                acc = 1.0 - (delta / target[i,0].cpu())
                acc_batch_sum += acc #np.clip(acc, 0, 1)
            
            acc_batch += (acc_batch_sum / BATCH_SIZE)
    acc = acc_batch / len(test_loader)
    print("epoch = %d   accuracy = %f" % (epoch, acc))
    return acc


# train
best_acc = 0
model_name = "resnext101_paper"
training_sequence = "final"
offset = 0
for epoch in range(1+offset, EPOCHS + 1+offset):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time + ": training epoch %2d" % (epoch))
    loss = train_one_epoch(net, DEVICE, train_loader, optimizer, epoch)
    acc = validate_model(net, DEVICE, test_loader, epoch)
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


