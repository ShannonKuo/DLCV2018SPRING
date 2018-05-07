import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from dataset import DatasetFromFolder
import scipy.misc
import os


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

num_epochs = 10
batch_size = 1
learning_rate = 1e-3
output_folder = './output'
test_output_folder = './test_output'
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#device = torch.device("cuda" if args.cuda else "cpu")

def load_image(folder):
    print("load_image...")
    x = []
    file_list = [file for file in os.listdir(folder) if file.endswith('.png')]
    file_list.sort()

    for i, file in enumerate(file_list):
        img = scipy.misc.imread(os.path.join(folder, file))
        x.append(img)
        #if (i > 10):
        #    break

    x = [transforms.ToTensor()(i) for i in x]
    dataloader = DataLoader(x, batch_size=batch_size, shuffle=True)
    print("finish load image")
    return dataloader


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def training(data_loader):
    print("start training")
    if args.cuda:
        model = autoencoder().cuda()
    else:
        model = autoencoder().cpu()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cnt = 0
    for epoch in range(num_epochs):
        for img in data_loader:
            if args.cuda:
                img = Variable(img).cuda()
            else:
                img = Variable(img).cpu()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            
            save_image(pic, output_folder + '/image_{}.png'.format(epoch))
     
    torch.save(model.state_dict(), './conv_autoencoder.pth')
    return model

def testing(model, data_loader):
    if not os.path.exists(output_folder):
        os.makedirs(test_output_folder)
    for i, img in enumerate(data_loader):
        if args.cuda:
            img = Variable(img).cuda()
        else:
            img = Variable(img).cpu()
        output = model(img)
        pic = to_img(output.cpu().data)
        save_image(pic, test_output_folder + '/image_{}.png'.format(i))
     

if __name__ == '__main__':
    train_data_loader = load_image('./hw4_data/train')
    test_data_loader = load_image('./hw4_data/test')
    model = training(train_data_loader)
    testing(model, test_data_loader)
