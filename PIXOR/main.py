import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from MultiLoss import MultiLoss
from PIXOR import Pixor
import os

# Load training and testing datasets from torchvision.datasets.Kitti
# Data structure of training is a list of 7481 * (image, label)
# Data structure of each image is 3-channel, 370 * 1224

train_data = datasets.Kitti(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.Kitti(root="data", train=False, download=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)

net = Pixor()
PATH = 'checkpoint/ckpt.pth'
if os.path.isdir('checkpoint'):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['net'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = MultiLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
epoches = 100
batch_size = 4


def train(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (input, label) in enumerate(trainloader):
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device)

        output = net(input)
        predict = output.premute([0, 2, 3, 1])
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

    print(f"Training {epoch + 1} done, correct={correct}, total={total}, loss={running_loss}")

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

    acc = 100. * correct / total
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
        # test(epoch)
        scheduler.step()
    print('Finished Training')