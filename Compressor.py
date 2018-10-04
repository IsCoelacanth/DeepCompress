import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
%matplotlib inline



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.STL10(root='./data', split='train',
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.STL10(root='./data', split='test',
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()

        self.E1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,3,padding=1),   # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)
        )

        self.E2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),  # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),  # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),  # batch x 64 x 7 x 7
                        nn.ReLU()
        )

    def forward(self, x):
        x = self.E1(x)
        x = self.E2(x)
        return x


class decoder(nn.Module):

    def __init__(self):
        super(decoder, self).__init__()

        self.D1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,128,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )

        self.D2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),
                        nn.ReLU()
        )

    def forward(self, x):
        x = self.D1(x)
        x = self.D2(x)
        return x

try:
    encoder, decoder = torch.load('./model/model.pkl')
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass


def train(en, de):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on: ", device)

    if not device == 'cpu':
        en.to(device)
        de.to(device)

    loss_plot = []
    epoch = 10
    lr = 0.0005

    params = list(en.parameters()) + list(de.parameters())
    loss_funtion = nn.BCELoss()
    opti = torch.optim.Adam(params, lr=lr)


    for i in range(epoch):
        loader_size = len(trainloader)
        accumilated_loss = 0
        for idx, data in enumerate(trainloader):
            img, _ = data
            loss = 0.0

            img_ = add_noise(img)
            if not device == 'cpu':
                img = img.to(device)
                img_ = img_.to(device)
            opti.zero_grad()

            oe = en(img_)
            od = de(oe)
            loss += loss_funtion(od, img)

            oe = en(img)
            od = de(oe)
            loss += loss_funtion(od, img)

            for i in range(10):
                oe = en(od)
                od = de(oe)
                loss += loss_funtion(od, img)
            
            accumilated_loss += loss.item()
            loss.backward()
            opti.step()
            
            if idx % 100 == 0:
                print('[idx = {}, current loss = {}', idx, loss.item())
                loss_plot.append(loss.cpu().numpy())
        print("epoch = {}, epoch_loss = {}", i, accumilated_loss/loader_size)
    torch.save([en, de], 'model/comp_decomp.pkl')


en = encoder()
de = decoder()
train(en, de)
