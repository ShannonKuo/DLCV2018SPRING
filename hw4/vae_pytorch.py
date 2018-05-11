import matplotlib
matplotlib.use('Agg')

import sys
import csv
import torch
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.manifold import TSNE
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
train = 1
if debug == 1:
    num_epochs = 3
else:
    num_epochs = 50
batch_size = 32
learning_rate = float(1e-5)
MSEloss = []
KLDloss = []
latent_spaces = []
nz = 512
ngf = 64
ndf = 64
nc = 3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
parser = argparse.ArgumentParser(description='VAE Example')

parser.add_argument('--lambdaKL', type=float, default=1e-5, metavar='N',
                    help='lambdaKL (default: 1e-5)')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

lambdaKL = args.lambdaKL
output_folder = './output_' + str(args.lambdaKL)
test_output_folder = './test_output_' + str(args.lambdaKL)
output_fig_folder = './output_fig_' + str(args.lambdaKL)

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
            #input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.fcn11 = nn.Linear(ndf * 8 * 4 * 4, 512)
        self.fcn12 = nn.Linear(ndf * 8 * 4 * 4, 512)
        self.fcn2 = nn.Linear(512, ndf * 8 * 4 * 4)
        
    def random_generate(self, z):
        z = self.fcn2(z)
        z = z.view(-1, ndf * 8, 4, 4)
        z = self.decoder(z)
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
        if training_testing == 'training':
            return eps.mul(std).add_(mu)
        else:
            for i in range(len(mu.data.cpu())):
                latent_spaces.append(mu.data.cpu().numpy()[i])
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.fcn2(z)
        z = z.view(-1, ndf * 8, 4, 4)
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
    KLD = torch.sum(KLD_element).mul_(-0.5)#.div_(batch_size)
    # KL divergence
    KLDloss.append(KLD.data[0])
    KLD = KLD.mul_(lambdaKL)
    MSEloss.append(MSE.data[0])
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

        random_generate_img(model)
        plot_loss()
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
    print("test_loss: ", test_loss)

def random_generate_img(model):
    model.eval()
    images = []
    if not os.path.exists(output_fig_folder):
        os.makedirs(output_fig_folder)

    for i in range(32):
        x = torch.randn(512)
        #r = np.random.normal(0.0, 1.0, 512)
        #print(r)

        #for i in range(512):
        #    x[i] = x[i] / 10
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
    t = np.arange(0.0, len(MSEloss), 1.0)
    fig.add_subplot(1, 2, 1)
    line, = plt.plot(t, MSEloss, lw=2)
    plt.xlabel('steps')
    plt.ylabel('MSE_loss')
    plt.title('MSE_loss vs steps')
    plt.ylim(0,0.5)

    t = np.arange(0.0, len(KLDloss), 1.0)
    fig.add_subplot(1, 2, 2)
    line, = plt.plot(t, KLDloss, lw=2)
    plt.xlabel('steps')
    plt.ylabel('KLD_loss')
    plt.title('KLD_loss vs steps')
    #plt.ylim(0,20000)

    plt.savefig(output_fig_folder + '/fig1_2.jpg')
    plt.close()

def calculate_tsne():
    color = []
    with open('./hw4_data/test.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            color.append(row['Male'])
            if debug == 1 and i > 10:
                break;
    color = np.array(color)
    print(len(latent_spaces))
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(latent_spaces)
    #ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE")
    plt.savefig(output_fig_folder + '/fig1_5.jpg')
    plt.close()


if __name__ == '__main__':
    if train == 1:
        torch.manual_seed(999)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(999)
        training_testing = 'training'
        train_data_loader, train_file_list = load_image('./hw4_data/train')
        model = training(train_data_loader, train_file_list)
    else:
        model = torch.load('./conv_autoencoder.pth')
    training_testing = 'testing'
    test_data_loader, test_file_list = load_image('./hw4_data/test')
    testing(model, test_data_loader, test_file_list)
    random_generate_img(model)
    plot_loss()
    calculate_tsne()
