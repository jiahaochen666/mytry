import torch
import torchvision
from VGG19 import VGG19
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch.nn as nn


net = VGG19()
PATH = 'checkpoint/ckpt.pth'
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['net'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )

testloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)

best_acc = 0
best_acc = checkpoint['acc']

def train(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print("Training %d done" % (epoch))

def test(epoch):
    global best_acc
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%, best: %f' % (
        100 * correct / total, best_acc))
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == '__main__':
    for epoch in range(100):  # loop over the dataset multiple times
        train(epoch)
        test(epoch)
        scheduler.step()
    print('Finished Training')