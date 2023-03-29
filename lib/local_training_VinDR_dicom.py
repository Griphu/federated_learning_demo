import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import wandb
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-pn','--project_name',default='VinDR Dicom 16b', required=True)
parser.add_argument('-rn','--run_name', default='png', required=True)
parser.add_argument('-d','--data_path', default = 'data/datasets/vindr_ds/images', required=True)
parser.add_argument('-m','--model_path', default = 'models/ddsm_four_classes_split.h5', required=True)
parser.add_argument('-c','--n_classes',default=4, required=True)
args = parser.parse_args()

wandb.init(project=args.project_name)
wandb.run.name=args.run_name
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("mps")


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.classifier = nn.Sequential(nn.Linear(in_features = 1000,out_features = 500),
                                 nn.ReLU(inplace = True),
                                 nn.Linear(in_features = 500,out_features = int(args.n_classes)))

# ct = 0
# for child in model.children():
#     ct += 1
#     if ct < 9:
#         for param in child.parameters():
#             param.requires_grad = False

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


def load_data():
    norm = [0.5, 0.5, 0.5]
    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm, norm)])

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm, norm)])

    train_data = ImageFolder(os.path.join(os.getenv('root_path'),args.data_path,"train"), transform=train_transform)
    valid_data = ImageFolder(os.path.join(os.getenv('root_path'),args.data_path,"valid"), transform=test_transform)

    return DataLoader(train_data, batch_size=12, shuffle=True), DataLoader(valid_data)


net = model.to(DEVICE)
trainloader, testloader = load_data()

# class local_training:
#     def __init__(self,net):
#         self.net = net

#     def fit_eval(self, epochs = 30):
#         for _ in range(epochs):
#             train(self.net, trainloader, epochs=1)
#             loss, accuracy = test(self.net,testloader)
#             wandb.log({'loss':loss, 'acc':accuracy})
#             print("loss:" + str(loss)+" acc: "+str(accuracy))
#     def save_model(self, path):
#         torch.save(self.net.state_dict(), f'{path}.h5')

# lt = local_training(net=net)

# lt.fit_eval(epochs = 10)

# lt.save_model(os.path.join(os.getenv('root_path'),args.model_path))


def train_model(net, num_epochs=10):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        train(net, trainloader, epochs=1)
        loss, accuracy = test(net,testloader)
        wandb.log({'loss':loss, 'acc':accuracy})
        print("loss:" + str(loss)+" acc: "+str(accuracy))

train_model(net, num_epochs=20)
torch.save(net.state_dict(), os.path.join(os.getenv('root_path'),args.model_path))