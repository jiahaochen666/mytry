import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from MultiLoss import MultiLoss
from myPixor import PIXOR
import os
from load_data import load_dataset
from eval import evaluate_model



PATH = 'checkpoint/ckpt.pth'
batch_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_loader = load_dataset(root="../../Data", batch_size=batch_size, device=device)
net = PIXOR().to()
best_loss = 10000
if os.path.isdir('checkpoint'):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
net.to(device)
criterion = MultiLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
epoches = 100


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def train(epoch):
    running_loss = 0.0
    printProgressBar(0, len(data_loader['train']), prefix='Progress:', suffix='Complete', length=50)
    for i, (input, label) in enumerate(data_loader['train']):
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device)

        output = net(input)
        predict = output.permute([0, 2, 3, 1])
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        printProgressBar(i, len(data_loader['train']), prefix = 'Progress:', suffix = 'Complete', length = 50)
    print(f"Epoch {epoch} Training Loss: {running_loss}")

def test(epoch):
    global best_loss
    running_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    printProgressBar(0, len(data_loader['val']), prefix='Progress:', suffix='Complete', length=50)
    with torch.no_grad():
        for i, data in enumerate(data_loader['val']):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            output = net(images)
            # the class with the highest energy is what we choose as prediction
            predict = output.permute([0, 2, 3, 1])
            loss = criterion(predict, labels)
            running_loss += loss.item()
            printProgressBar(i, len(data_loader['val']), prefix='Progress:', suffix='Complete', length=50)

    print(f"Epoch {epoch} Testing Loss: {running_loss}, Best: {best_loss}")


    if running_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': running_loss,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ckpt.pth')
        best_loss = running_loss

if __name__ == '__main__':
    torch.cuda.empty_cache()
    for epoch in range(epoches):  # loop over the dataset multiple times
        train(epoch)
        test(epoch)
        scheduler.step()
    print('Finished Training')