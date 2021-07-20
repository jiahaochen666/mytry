import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from MultiLoss import MultiLoss
from myPixor import PIXOR
import os
from load_data import load_dataset



PATH = 'checkpoint/ckpt.pth'
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_loader = load_dataset(root="../../Data", batch_size=batch_size, device=device)
net = PIXOR().to()
if os.path.isdir('checkpoint'):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['net'])
net.to(device)
criterion = MultiLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
epoches = 100


def train(input, label):
    running_loss = 0.0
    optimizer.zero_grad()
    input, label = input.to(device), label.to(device)

    output = net(input)
    predict = output.permute([0, 2, 3, 1])
    loss = criterion(predict, label)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    return running_loss

def test():
    global best_acc
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader['val']:
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
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_acc = acc

if __name__ == '__main__':
    torch.cuda.empty_cache()
    for epoch in range(100):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, (input, label) in enumerate(data_loader['train']):
            running_loss += train(input, label)
            if i % 25 == 0:
                print(f"Training round {i % 25}, Loss: {running_loss}")
                running_loss = 0.0
            if i % 100 == 0:
                print("Testing...")
                test()
            scheduler.step()
    print('Finished Training')