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
import sys
import numpy as np
import matplotlib.pyplot as plt


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

debug = 1
if debug == 1:
    num_epochs = 3
else:
    num_epochs = 30
batch_size = 32
learning_rate = 1e-5

nz = 100
ngf = 64
ndf = 64
nc = 3

training_testing = sys.argv[1]
dataset_folder = sys.argv[2]
output_folder = sys.argv[3]
all_loss_G = []
all_loss_D = []
all_accuracy = []
if torch.cuda.is_available():
    cuda = 1
else:
    cuda = 0
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class GAN_generator(nn.Module):
    def __init__(self):
        super(GAN_generator, self).__init__()
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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


class GAN_discriminator(nn.Module):
    def __init__(self):
        super(GAN_discriminator, self).__init__()
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
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.discriminator(x)
        return x.view(-1, 1).squeeze(1)

def training(data_loader, file_list):
    print("start training")
    if cuda == 1:
        generator = GAN_generator().cuda()
    else:
        generator = GAN_generator().cpu()
    generator.apply(weights_init)

    if cuda == 1:
        discriminator = GAN_discriminator().cuda()
    else:
        discriminator = GAN_discriminator().cpu()
    discriminator.apply(weights_init)

    generator.train()
    discriminator.train()

    optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5,0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5,0.999))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_loss = []
    for epoch in range(num_epochs):
        for i, img in enumerate(data_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            #log(D(x))
            discriminator.zero_grad()

            if cuda == 1:
                img = Variable(img).cuda()
            else:
                img = Variable(img).cpu()
            real_predict = discriminator(img)
            vector_size = real_predict.shape[0]

            if cuda == 1:
                real_label = Variable(torch.ones(vector_size)).cuda()
            else:
                real_label = Variable(torch.ones(vector_size)).cpu()
            lossD_real = nn.BCELoss()(real_predict, real_label)
            D_x = real_predict.mean().data[0]

            #log(1-D(G(z)))
            if cuda == 1:
                noise = Variable(torch.randn(vector_size, nz, 1, 1)).cuda()
            else:
                noise = Variable(torch.randn(vector_size, nz, 1, 1)).cpu()

            fake_img = generator(noise)
            fake_predict = discriminator(fake_img.detach())

            if cuda == 1:
                fake_label = Variable(torch.zeros(vector_size)).cuda()
            else:
                fake_label = Variable(torch.zeros(vector_size)).cpu()

            lossD_fake = nn.BCELoss()(fake_predict, fake_label)
            D_G_z1 = fake_predict.mean().data[0]
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            file_D = open('./gan_lossD.txt', "a+")
            file_D.write(str(lossD.data[0]))
            file_D.write('\n')
            file_D.close()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            output = discriminator(fake_img) 

            lossG = nn.BCELoss()(output, real_label)
            lossG.backward()
            D_G_z2 = output.mean().data[0]
            optimizerG.step()

            file_G = open('./gan_lossG.txt', "a+")
            file_G.write(str(lossG.data[0]))
            file_G.write('\n')
            file_G.close()

            # ===================log========================
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                 % (epoch + 1, num_epochs, i, len(data_loader),
                 lossD.data[0], lossG.data[0], D_x, D_G_z1, D_G_z2))

        torch.save(generator.state_dict(), './gan_generator.pth')
        torch.save(discriminator.state_dict(), './gan_discriminator.pth')
    return generator, discriminator

def plot_loss():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_D = open('./gan_lossD.txt')
    for line in file_D:
        all_loss_D.append(float(line))
    file_D.close()

    file_G = open('./gan_lossG.txt')
    for line in file_G:
        all_loss_G.append(float(line))
    file_G.close()
    
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
    #if cuda == 1:
    #    noise = Variable(torch.randn(32, nz, 1, 1)).cuda()
    #else:
    noise = Variable(torch.randn(32, nz, 1, 1)).cpu()
    fake_img = generator(noise)
    pic = to_img(fake_img.cpu().data)
    out = torchvision.utils.make_grid(pic, nrow=8)
    save_image(out, output_folder + '/fig2_3.jpg', normalize=True)

if __name__ == '__main__':
    if cuda == 1:
        torch.cuda.manual_seed_all(999)
    else:
        torch.manual_seed(999)
    if training_testing == 'train':
        train_data_loader, train_file_list = load_image(dataset_folder + '/train')
        model_G, model_D = training(train_data_loader, train_file_list)
    elif training_testing == 'test':
        model_G = GAN_generator()
        model_D = GAN_discriminator()
        model_G.load_state_dict(torch.load('./gan_generator.pth'))
        model_D.load_state_dict(torch.load('./gan_discriminator.pth'))
        plot_loss()
        generate_img(model_G)
