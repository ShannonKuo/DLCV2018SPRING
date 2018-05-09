import matplotlib
matplotlib.use('Agg')
import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F
import scipy.misc
import os
import numpy as np
import matplotlib.pyplot as plt
#print(torch.__version__)


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

debug = 0
if debug == 1:
    num_epochs = 3
else:
    num_epochs = 30
batch_size = 32
learning_rate = 1e-5
output_folder = './output'
test_output_folder = './test_output'
output_fig_folder = './output_fig'
MSEloss = []
KLDloss = []

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
        if (i > 10 and debug == 1):
           break

    x = [img_transform(i) for i in x]
    dataloader = DataLoader(x, batch_size=batch_size, shuffle=False)
    print("finish load image")
    return (dataloader, file_list)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=2),  # b, 16, 64, 64
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 32, 32
            nn.Conv2d(16, 4, 5, stride=1, padding=2),  # b, 8, 32, 32
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 4, 16, 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=2),  # b, 8, 16, 16
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=0),  # b, 16, 32, 32
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 3, 5, stride=1, padding=0),  # b, 3, 64, 64
            nn.Tanh()
        )
        self.fcn11 = nn.Linear(1024, 1024)
        self.fcn12 = nn.Linear(1024, 1024)
        #self.conv11 = nn.Conv2d(8, 4, 3, stride=2, padding=2) # 4, 8, 8
        #self.conv12 = nn.Conv2d(8, 4, 3, stride=2, padding=2)
    def random_generate(self, vector):
        vector = vector.view(-1, 4, 16, 16)
        z = self.decoder(vector)
        return z
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])

        return self.fcn11(x), self.fcn12(x)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        if self.training:
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = z.view(-1, 4, 16, 16)
        z = self.decoder(z)
        return z, mu, logvar
        #x = self.encoder(x)
        #x = self.conv11(x)
        #x = self.decoder(x)
        #return x
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    criterion = nn.MSELoss()
    MSE = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5).div_(batch_size)
    # KL divergence
    lambdaKL = 1e-6
    KLD = KLD.mul_(lambdaKL)
    MSEloss.append(MSE.data[0])
    KLDloss.append(KLD.data[0])
    return MSE + KLD

def training(data_loader, file_list):
    print("start training")
    if args.cuda:
        model = autoencoder().cuda()
    else:
        model = autoencoder().cpu()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    all_loss = []
    for epoch in range(num_epochs):
        idx = 0
        train_loss = 0.0
        for img in data_loader:
            if args.cuda:
                img = Variable(img).cuda()
            else:
                img = Variable(img).cpu()
            img_show = to_img(img.cpu().data)
            # ===================forward=====================
            output, mu, logvar = model(img)
            loss = loss_function(output, img, mu, logvar)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()

            if idx < 10:
                pic = to_img(output.cpu().data)
                for i in range(len(pic)):
                    file_path = output_folder + '/' + file_list[idx] 
                    save_image(pic[i], output_folder + '/' + file_list[idx], normalize=True)
                    idx += 1
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
    torch.save(model.state_dict(), './conv_autoencoder.pth')
    return model

def testing(model, data_loader, file_list):
    print("start test...")
    model.eval()
    if not os.path.exists(test_output_folder):
        os.makedirs(test_output_folder)
    if not os.path.exists(output_fig_folder):
        os.makedirs(output_fig_folder)
    idx = 0
    test_loss = 0.0
    ten_images = [0 for x in range(20)]
    choose_cnt = 0
    for i, img in enumerate(data_loader):
        if args.cuda:
            img = Variable(img).cuda()
        else:
            img = Variable(img).cpu()
        output, mu, logvar = model(img)
        loss = nn.MSELoss()(output, img)
        test_loss += loss.data[0] / len(data_loader)

        pic = to_img(output.cpu().data)
        input = to_img(img.cpu().data)
        for j in range(len(pic)):
            file_path = test_output_folder + '/' + file_list[idx]
            save_image(pic[j], test_output_folder + '/' + file_list[idx], normalize=True)
            idx += 1
            if choose_cnt < 10:
                ten_images[choose_cnt] = input[j]
                ten_images[10 + choose_cnt] = pic[j]
                choose_cnt += 1

    out = torchvision.utils.make_grid(ten_images, nrow=10)
    save_image(out, output_fig_folder + '/fig1_3.jpg', normalize=True)

def random_generate_img(model):
    model.eval()
    images = []
    if not os.path.exists(output_fig_folder):
        os.makedirs(output_fig_folder)

    for i in range(32):
        x = torch.randn(1024)

        if args.cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x).cpu()
        output = model.random_generate(x)
        pic = to_img(output.cpu().data)
        images.append(pic[0])
    out = torchvision.utils.make_grid(images, nrow=8)
    save_image(out, output_fig_folder + '/fig1_4.jpg', normalize=True)


def plot_loss():
    if not os.path.exists(output_fig_folder):
        os.makedirs(output_fig_folder)

    fig=plt.figure(figsize=(15, 5))
    t = np.arange(0.0, num_epochs, 1.0)
    fig.add_subplot(1, 2, 1)
    line, = plt.plot(t, MSEloss, lw=2)
    plt.xlabel('epochs')
    plt.ylabel('MSE_loss')
    plt.title('MSE_loss vs epochs')
    plt.ylim(-2,2)

    t = np.arange(0.0, num_epochs, 1.0)
    fig.add_subplot(1, 2, 2)
    line, = plt.plot(t, KLDloss, lw=2)
    plt.xlabel('epochs')
    plt.ylabel('KLD_loss')
    plt.title('KDL_loss vs epochs')
    plt.ylim(-2,2)

    plt.savefig(output_fig_folder + '/fig1_2.jpg')
    plt.close()

if __name__ == '__main__':
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
    train_data_loader, train_file_list = load_image('./hw4_data/train')
    test_data_loader, test_file_list = load_image('./hw4_data/test')
    model = training(train_data_loader, train_file_list)
    testing(model, test_data_loader, test_file_list)
    random_generate_img(model)
    plot_loss()
