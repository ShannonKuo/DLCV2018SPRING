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
training_testing = sys.argv[1]
if debug == 1:
    num_epochs = 3
else:
    num_epochs = 40
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

lambdaKL = 1e-5
dataset_folder = sys.argv[2]
output_fig_folder = sys.argv[3]

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
        if training_testing == 'train':
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

def loss_function(recon_x, x, mu, logvar):
    criterion = nn.MSELoss()
    MSE = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    KLDloss.append(KLD.data[0])
    KLD = KLD.mul_(lambdaKL)
    MSEloss.append(MSE.data[0])

    file = open('./vae_KLDloss.txt', "a+")
    file.write(str(KLD.data[0]))
    file.write('\n')
    file.close()
    
    file = open('./vae_MSEloss.txt', "a+")
    file.write(str(MSE.data[0]))
    file.write('\n')
    file.close()
    return MSE + KLD

def training(data_loader, file_list):
    print("start training")
    if torch.cuda.is_available():
        model = autoencoder().cuda()
    else:
        model = autoencoder().cpu()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

    all_loss = []
    for epoch in range(num_epochs):
        idx = 0
        train_loss = 0.0
        for img in data_loader:
            if torch.cuda.is_available():
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
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.
                format(epoch+1, num_epochs, train_loss))
        torch.save(model.state_dict(), './conv_autoencoder.pth')
    return model

def testing(model, data_loader, file_list):
    print("start test...")
    model.eval()
    if not os.path.exists(output_fig_folder):
        os.makedirs(output_fig_folder)
    idx = 0
    test_loss = 0.0
    ten_images = [0 for x in range(20)]
    choose_cnt = 0
    for i, img in enumerate(data_loader):
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = Variable(img).cpu()
        output, mu, logvar = model(img)
        loss = nn.MSELoss()(output, img)
        test_loss += loss.data[0]

        pic = to_img(output.cpu().data)
        input = to_img(img.cpu().data)
        for j in range(len(pic)):
            idx += 1
            if choose_cnt < 10:
                ten_images[choose_cnt] = input[j]
                ten_images[10 + choose_cnt] = pic[j]
                choose_cnt += 1

    out = torchvision.utils.make_grid(ten_images, nrow=10)
    save_image(out, output_fig_folder + '/fig1_3.jpg', normalize=True)
    print("test_loss: ", test_loss)

def random_generate_img(model):
    print("random generate image...")
    model.eval()
    images = []
    if not os.path.exists(output_fig_folder):
        os.makedirs(output_fig_folder)

    for i in range(32):
        x = torch.randn(512)
        if torch.cuda.is_available():
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
    
    file_KLD = open('./vae_KLDloss.txt')
    for line in file_KLD:
        KLDloss.append(float(line) * 100000)
    file_KLD.close()

    file_MSE = open('./vae_MSEloss.txt')
    for line in file_MSE:
        MSEloss.append(float(line))
    file_MSE.close()
    
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

    plt.savefig(output_fig_folder + '/fig1_2.jpg')
    plt.close()

def calculate_tsne():
    color = []
    with open(dataset_folder + '/test.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            color.append(row['Male'])
            if debug == 1 and i > 10:
                break;
    color = np.array(color)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(latent_spaces)
    plt.scatter(Y[:, 0], Y[:, 1], s=5, c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE")
    plt.savefig(output_fig_folder + '/fig1_5.jpg')
    plt.close()


if __name__ == '__main__':

    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    if training_testing == 'train':
        train_data_loader, train_file_list = load_image(dataset_folder + '/train')
        model = training(train_data_loader, train_file_list)
    elif training_testing == 'test':
        if torch.cuda.is_available():
            model = autoencoder().cuda()
        else:
            model = autoencoder().cpu()
        model.load_state_dict(torch.load('./conv_autoencoder.pth'))
        test_data_loader, test_file_list = load_image(dataset_folder + '/test')
        testing(model, test_data_loader, test_file_list)
        random_generate_img(model)
        plot_loss()
        calculate_tsne()
