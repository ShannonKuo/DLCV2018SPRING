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
import csv
import numpy as np
import matplotlib.pyplot as plt


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

debug = 1
train = 1
attributeID = 8
if debug == 1:
    num_epochs = 3
else:
    num_epochs = 50
batch_size = 32
learning_rate = 1e-4

nz = 100
nl = 1
ngf = 64
ndf = 64
nc = 3
output_folder = './output_acgan'
all_loss_G = []
all_loss_D = []
all_accuracy = []

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
parser = argparse.ArgumentParser(description='ACGAN')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()


def load_image(folder, csv_path):
    print("load_image...")
    x = []
    label = []
    file_list = [file for file in os.listdir(folder) if file.endswith('.png')]
    file_list.sort()

    for i, file in enumerate(file_list):
        img = scipy.misc.imread(os.path.join(folder, file))
        x.append(img)
        if (i > 10 and debug == 1):
           break
    label = np.genfromtxt(csv_path, delimiter=',', dtype=float)
    label = label[1:, attributeID: attributeID + 1]
    if debug == 1:
        label = label[0: 12, :]
    label = torch.from_numpy(label).type(torch.FloatTensor)
    data = [(img_transform(x[i]), label[i]) for i in range(len(x))]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    print("finish load image")
    return (dataloader, file_list)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ACGAN_generator(nn.Module):
    def __init__(self):
        super(ACGAN_generator, self).__init__()
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nl, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
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

    def forward(self, x):
        output = self.generator(x)
        return output


class ACGAN_discriminator(nn.Module):
    def __init__(self):
        super(ACGAN_discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.fc1 = nn.Linear(ndf * 8 * 4 * 4, 1)
        self.fc2 = nn.Linear(ndf * 8 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.discriminator(x)
        x = x.view(-1, ndf * 8 * 4 * 4)
        dis = self.sigmoid(self.fc1(x))
        aux = self.sigmoid(self.fc2(x))
        return dis, aux

def training(data_loader, file_list):
    print("start training")
    if args.cuda:
        generator = ACGAN_generator().cuda()
    else:
        generator = ACGAN_generator().cpu()
    generator.apply(weights_init)

    if args.cuda:
        discriminator = ACGAN_discriminator().cuda()
    else:
        discriminator = ACGAN_discriminator().cpu()
    discriminator.apply(weights_init)

    generator.train()
    discriminator.train()
    optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5,0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5,0.999))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_loss = []
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            #log(D(x))
            discriminator.zero_grad()
            img = data[0]
            aux_label = data[1]
            print(aux_label.shape)
            if args.cuda:
                img = Variable(img).cuda()
                aux_label = Variable(aux_label).cuda()
            else:
                img = Variable(img).cpu()
                aux_label = Variable(aux_label).cpu()

            dis_real_predict, aux_real_predict = discriminator(img)
            vector_size = dis_real_predict.shape[0]

            if args.cuda:
                dis_real_label = Variable(torch.ones(vector_size)).cuda()
            else:
                dis_real_label = Variable(torch.ones(vector_size)).cpu()
            dis_lossD_real = nn.BCELoss()(dis_real_predict, dis_real_label)
            aux_lossD_real = nn.BCELoss()(aux_real_predict, aux_label)
            lossD_real = dis_lossD_real + aux_lossD_real
            D_x = dis_real_predict.mean().data[0]

            #log(1-D(G(z)))
            noise = torch.randn(vector_size, nz, 1, 1)
            random_aux = np.random.randint(2, size=(vector_size, nl, 1, 1))
            random_aux = torch.from_numpy(random_aux).type(torch.FloatTensor)
            noise = torch.cat((noise, random_aux), dim=1)
            if args.cuda:
                noise = Variable(noise).cuda()
            else:
                noise = Variable(noise).cpu()
            fake_img = generator(noise)
            dis_fake_predict, aux_fake_predict = discriminator(fake_img.detach())

            if args.cuda:
                dis_fake_label = Variable(torch.zeros(vector_size)).cuda()
            else:
                dis_fake_label = Variable(torch.zeros(vector_size)).cpu()

            dis_lossD_fake = nn.BCELoss()(dis_fake_predict, dis_fake_label)
            aux_lossD_fake = nn.BCELoss()(aux_fake_predict, aux_label)
            D_G_z1 = dis_fake_predict.mean().data[0]
            lossD_fake = dis_lossD_fake + aux_lossD_fake

            lossD = lossD_real + lossD_fake
            lossD.backward()
            all_loss_D.append(lossD.data[0])
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            output, output_aux = discriminator(fake_img) 

            dis_lossG = nn.BCELoss()(output, dis_real_label)
            aux_lossG = nn.BCELoss()(output_aux, aux_label)
            lossG = dis_lossG + aux_lossG
            lossG.backward()
            all_loss_G.append(lossG.data[0])
            D_G_z2 = output.mean().data[0]
            optimizerG.step()

            # ===================log========================
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                 % (epoch + 1, num_epochs, i, len(data_loader),
                 lossD.data[0], lossG.data[0], D_x, D_G_z1, D_G_z2))

        generate_img(generator)
        plot_loss()

    torch.save(generator.state_dict(), './acgan_generator.pth')
    torch.save(discriminator.state_dict(), './acgan_discriminator.pth')
    return generator, discriminator

def plot_loss():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fig=plt.figure(figsize=(15, 5))
    t = np.arange(0.0, len(all_loss_D), 1.0)
    fig.add_subplot(1, 2, 1)
    line, = plt.plot(t, all_loss_D, lw=2)
    plt.xlabel('steps')
    plt.ylabel('Discriminator_loss')
    plt.title('Discriminator_loss vs steps')

    t = np.arange(0.0, len(all_loss_G), 1.0)
    fig.add_subplot(1, 2, 2)
    line, = plt.plot(t, all_loss_G, lw=2)
    plt.xlabel('steps')
    plt.ylabel('Generator_loss')
    plt.title('Generator_loss vs steps')

    plt.savefig(output_folder + '/fig2_2.jpg')
    plt.close()

def generate_img(generator):
    noise = torch.randn(10, nz, 1, 1)
    noise = torch.cat((noise, noise), dim=0)
    random_aux = np.zeros((10, 1, 1, 1))
    random_aux2 = np.ones((10, 1, 1, 1))
    random_aux = np.vstack((random_aux, random_aux2))
    random_aux = torch.from_numpy(random_aux).type(torch.FloatTensor)
    noise = torch.cat((noise, random_aux), dim=1)
    if args.cuda:
        noise = Variable(noise).cuda()
    else:
        noise = Variable(noise).cpu()
    fake_img = generator(noise)
    pic = to_img(fake_img.cpu().data)
    out = torchvision.utils.make_grid(pic, nrow=10)
    save_image(out, output_folder + '/fig2_3.jpg', normalize=True)


if __name__ == '__main__':
    if train == 1:
        np.random.seed(999)
        torch.manual_seed(999)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(999)
        train_data_loader, train_file_list = load_image('./hw4_data/train', './hw4_data/train.csv')
        model_G, model_D = training(train_data_loader, train_file_list)
    else:
        model_G = torch.load('./acgan_generator.pth')
        model_D = torch.load('./acgan_discriminator.pth')
    plot_loss()
    generate_img(model_G)
