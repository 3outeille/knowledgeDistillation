import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import shutil
import numpy as np
import random

from distillation.hintonDistiller import HintonDistiller
from distillation.utils import MLP, PseudoDataset

def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_mnist(BATCH_SIZE=64, download=True):
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])])

    train_dataset = datasets.MNIST('./data', train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=download, transform=transform)

    # Split dataset into training set and validation set.
    train_dataset, val_dataset = random_split(train_dataset, (55000, 5000))

    print("Image Shape: {}".format(train_dataset[0][0].numpy().shape), end = '\n\n')
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)))
    print("Test Set:       {} samples".format(len(test_dataset)))

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    return (train_loader, val_loader, test_loader)

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP(nn.Module):
    """
    Simple MLP model for projector and predictor in BYOL paper.
    
    :param inputDim: int; amount of input nodes
    :param projectionDim: int; amount of output nodes
    :param hiddenDim: int; amount of hidden nodes
    """
    def __init__(self, inputDim, projectionDim, hiddenDim=4096):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(inputDim, hiddenDim)
        self.bn = nn.BatchNorm1d(hiddenDim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hiddenDim, projectionDim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

def train(train_loader, val_loader):
    set_all_seeds(42)
    
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = TeacherNet()
    student = StudentNet()
    distiller = HintonDistiller(alpha=0.5, studentLayer=-1, teacherLayer=-1)

    # Initialize objectives and optimizer
    objective = nn.CrossEntropyLoss()
    distillObjective = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    # Load state if checkpoint is provided
    checkpoint = "teacher.pth"
    teacher.load_state_dict(torch.load(checkpoint))
    startEpoch, epochs = 1, 5

    # Construct tensorboard logger
    distiller.init_tensorboard_logger()

    for epoch in range(startEpoch, epochs+1):
        # Training step for one full epoch
        trainMetrics = distiller.train_step(student=student,
                                            teacher=teacher,
                                            dataloader=train_loader,
                                            optimizer=optimizer,
                                            objective=objective,
                                            distillObjective=distillObjective)
        
        # Validation step for one full epoch
        validMetrics = distiller.validate(student=student,
                                        dataloader=val_loader,
                                        objective=objective)
        metrics = {**trainMetrics, **validMetrics}
        
        # Log to tensorbard
        distiller.log(epoch, metrics)

        # Save model
        distiller.save(epoch, student, teacher, optimizer)
        
        # Print epoch performance
        distiller.print_epoch(epoch, epochs, metrics)

    torch.save(student.state_dict(), "./student.pth")

def eval(test_loader, modelname="teacher"):
    if modelname == "teacher":
        model = TeacherNet()
        model.load_state_dict(torch.load("./teacher.pth"))
    else:
        model = StudentNet()
        model.load_state_dict(torch.load("./student.pth"))

    distiller = HintonDistiller(alpha=0.5, studentLayer=-1, teacherLayer=-1)
    objective = nn.CrossEntropyLoss()

    validMetrics = distiller.validate(student=model,
                                    dataloader=test_loader,
                                    objective=objective)
    print(validMetrics)

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_mnist(download=False)

    train(train_loader, val_loader)
    eval(test_loader, modelname="student")
