import torch
import torch.optim as optim
from MultiLoss import MultiLoss
from myPixor import PIXOR
import os
from load_data import load_dataset
import numpy as np

#################################################
######### Configure here when you start #########
#################################################
#Todo
start = 107
PATH = '../../../../opod/ningfei/checkpoint'
ROOT = "../../../../opod/ningfei/pixor_pytorch/KITTI/"
batch_size = 4
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_loader = load_dataset(root=ROOT, batch_size=batch_size, device=device)
net = PIXOR().to(device)
best_loss = 10000
best_epoch = 0
epoches = 25
if os.path.isdir(PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start = checkpoint['epoch']
    best_epoch = checkpoint['best']
net.to(device)
criterion = MultiLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

###########################################################################

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='|', printEnd="\r"):
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
    # print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    print(prefix + " |" + bar + "| " + percent + "% " + suffix, end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def train(epoch):
    running_loss = 0.0
    printProgressBar(0, len(data_loader['train']), prefix='Progress:', suffix='Complete', length=50)
    for i, (input, label) in enumerate(data_loader['train']):
        optimizer.zero_grad()
        input, label = input.to(device), label.to(device)

        output = net(input)
        predict = output.permute([0, 2, 3, 1])
        loss = criterion(predict, label)
        if not np.isnan(loss.item()) and not np.isinf(loss.item()):
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        printProgressBar(i, len(data_loader['train']), prefix='Progress:', suffix='Complete', length=50)
    # print(f"Epoch {epoch} Training Loss: {running_loss}")
    print("Epoch " + str(epoch) + " Trainling Loss: " + str(running_loss))


def test(epoch):
    global best_loss
    global best_epoch
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
            if not np.isnan(loss.item()):
                running_loss += loss.item()
            printProgressBar(i, len(data_loader['val']), prefix='Progress:', suffix='Complete', length=50)

    # print(f"Epoch {epoch} Testing Loss: {running_loss}, Best: {best_loss}")
    print("Epoch " + str(epoch) + " Testing Loss: " + str(running_loss) + ", Best: " + str(best_loss))

    print('Saving..')
    if running_loss < best_loss:
        best_epoch = epoch
        best_loss = running_loss
    state = {
        'epoch': epoch,
        'net': net.state_dict(),
        'loss': running_loss,
        'best': best_epoch
    }

    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    torch.save(state, PATH + '/ckpt' + str(epoch) + '.pth')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    for epoch in range(start + 1, start + 1 + epoches):  # loop over the dataset multiple times
        print("Starting Epoch: " + str(epoch))
        train(epoch)
        test(epoch)
        scheduler.step()
    print('Finished Training')
