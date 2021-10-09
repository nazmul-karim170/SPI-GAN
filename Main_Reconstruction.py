#!/usr/bin/env python

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data import DataLoader
from math import log10
import pandas as pd 

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model_final import Generator, Discriminator, FeatureExtractor, PreGenerator
from tensorboard_logger import configure, log_value
from tqdm import tqdm
import pytorch_ssim

from utils import Visualizer
from utils_cs import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform


        ### Initialize External Variables ### 
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--dataset', type=str, default='stl10', help='cifar10 | cifar100 | folder')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=400, type=int, help='train epoch number')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--generatorLR', type=float, default=0.00008, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true',default= True, help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints/For_paper', help='folder to output model checkpoints')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ## Training and Validation Set ###
train_set = TrainDatasetFromFolder('data/Reconstruction/STL_train_0.20_100k.npy', 'data/Reconstruction/STL_train.npy', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set   = ValDatasetFromFolder('data/Reconstruction/STL_test_0.20_100k.npy'   , 'data/Reconstruction/STL_test.npy', crop_size=CROP_SIZE,  upscale_factor=UPSCALE_FACTOR)

train_loader     = DataLoader(dataset=train_set, num_workers=8, batch_size=64, shuffle=True, drop_last = True)
val_loader       = DataLoader(dataset=val_set, num_workers=8, batch_size=64, shuffle=False, drop_last = True)
val_loader_pre   = DataLoader(dataset=val_set, num_workers=8, batch_size=64, shuffle=False, drop_last = True)

    ### Get the generator ###
generator       = Generator(14, opt.upscale_factor)

    ### Saving File ###
num_first_epoch = 0
num_pre_epoch   = 15
num_fin_epoch   = 200 

extension  = '_100k'
extension1 = '_100k'
save_file_first = 'statistics_first'+ extension1 +'.csv'
save_file_pre   = 'statistics_pretrain'+ extension1 +'.csv'
save_file_final = 'statistics_final'+ extension1 +'.csv'

model_file_first = '%s/First_block_STL'+ extension +'.pth'
model_file_pre   = '%s/Pretrain_STL'+ extension +'.pth'
model_file_final = '%s/Final_block_STL'+ extension +'.pth'
model_file_discriminator = '%s/discriminator_STL'+ extension +'.pth'

if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights))

    ##### Get the discriminator ##
discriminator = Discriminator()
if opt.discriminatorWeights != '':
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))


    #### For the content loss ##
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# Loss Functions
content_criterion     = nn.MSELoss()
adversarial_criterion = nn.BCELoss()
tv_criterion          = TVLoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# If gpu is to be used
device = 'cuda:1'
if opt.cuda:
    generator.to(device)
    discriminator.to(device)
    feature_extractor.to(device)
    content_criterion.to(device)
    adversarial_criterion.to(device)
    ones_const = ones_const.to(device)
    tv_criterion.to(device)


optim_generator     = optim.Adam(generator.parameters(), lr=4*opt.generatorLR)
# optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

configure('logs/' + opt.dataset + '-' + str(opt.batchSize) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR), flush_secs=5)
visualizer = Visualizer(image_size=CROP_SIZE*CROP_SIZE)


best_psnr = best_ssim = 0
print('.......................Generator Pre-training........')
results = {'psnr': [], 'ssim': []}


        #### If the generator is already pre-trained set gen_pretrain to "True" ####
gen_pretrain = False
if gen_pretrain:
    generator.load_state_dict(torch.load(model_file_pre % opt.out))

else:
        #### Pre-train generator using raw MSE loss  ####
    print('Generator pre-training')
    for epoch in range(num_pre_epoch):
        train_bar = tqdm(train_loader)
        mean_generator_content_loss = 0.0
        generator.train()
        batch_num = 0
        for data, target in train_bar:
            
            ## Generate data
            high_res_real = target
            low_res = data

            ## Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(high_res_real.to(device))
                high_res_fake = generator(Variable(low_res).to(device))
            else:
                high_res_real = Variable(high_res_real)
                high_res_fake = generator(Variable(low_res))

            ######### Train generator #########
            generator.zero_grad()

            generator_content_loss      = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.item()

            generator_content_loss.backward()
            optim_generator.step()
            batch_num += 1

            ######### Status and display #########
            sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, num_pre_epoch, batch_num, len(train_loader), generator_content_loss.item()))
            visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, num_pre_epoch, batch_num, len(train_loader), mean_generator_content_loss/len(train_loader)))
        log_value('generator_mse_loss', mean_generator_content_loss/len(train_loader), epoch)

            ### Do checkpointing  ##
        # torch.save(generator.state_dict(), model_file_pre % opt.out)
            
            #### Validate the generator ###
        generator.eval()
        out_path = 'training_results/STL/pre_train/ratio_0.30/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with torch.no_grad():
            val_bar = tqdm(val_loader_pre)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:   ## We probably dont need the val_hr_restore image 
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr

                if torch.cuda.is_available():
                    lr = lr.to(device)
                    hr = hr.to(device)
                
                sr = generator(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to HR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                    ## For Saving the Images ## 
            #     val_images.extend(
            #         [display_transform()(lr.data.cpu().squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
            #          display_transform()(sr.data.cpu().squeeze(0))])
            
            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images[0:5], desc='[saving training results]')
            # index = 1

            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     index += 1

            cur_psnr = valing_results['psnr']
            cur_ssim = valing_results['ssim']
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])

            ## For the best PSNR, do checkpointing
            if cur_psnr>best_psnr:
                best_ssim = cur_ssim
                best_psnr = cur_psnr
                
                torch.save(generator.state_dict(), model_file_pre % opt.out)
                
            out_path = 'statistics/For_Paper/'
            data_frame = pd.DataFrame(
                data={'Loss_D': 0, 'Loss_G': 0, 'Score_D': 0,
                      'Score_G': 0, 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(epoch + 1))

            data_frame.to_csv(out_path + save_file_pre, index_label='Epoch')


### GAN Training 
optim_generator_final = optim.Adam(generator.parameters(),  lr=opt.generatorLR*0.1, weight_decay = 0.005)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR*0.1, weight_decay = 0.005)


best_psnr = best_ssim = 0
results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
print('SRGAN training')

for epoch in range(num_fin_epoch):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    discriminator.train()
    generator.train()
    batch_num = 0

    for data, target in train_bar:

        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size
        
        #### Generate data
        high_res_real = target
        low_res       = data


        #### Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.to(device))
            high_res_fake = generator(Variable(low_res).to(device))
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).to(device)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).to(device)

        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)
        
                ######### Train Discriminator #########
        discriminator.zero_grad()
        real_out= discriminator(high_res_real)
        fake_out = discriminator(Variable(high_res_fake.data))
        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)

        mean_discriminator_loss += discriminator_loss.item()
        
        discriminator_loss.backward()
        optim_discriminator.step()

                ######### Train generator #########
        generator.zero_grad()

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

                ### Generator Losses (One can also add TV loss) ## 
        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.item()
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        generator_tv_loss  = tv_criterion(high_res_fake)
        mean_generator_adversarial_loss += generator_adversarial_loss.item()

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss + 2e-8*generator_tv_loss
        mean_generator_total_loss += generator_total_loss.item()
        
        generator_total_loss.backward()
        optim_generator_final.step()

        batch_num += 1    
        
                ######### Status and Display #########
            ## loss for current batch before optimization 
        running_results['g_loss'] += generator_total_loss.item() * batch_size
        running_results['d_loss'] += discriminator_loss.item() * batch_size
        running_results['d_score'] += torch.mean(real_out).item() * batch_size
        running_results['g_score'] += torch.mean(fake_out).item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))
    #     sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch, opt.num_epochs , batch_num, len(train_loader),
    #     discriminator_loss.item(), generator_content_loss.item(), generator_adversarial_loss.item(), generator_total_loss.item()))
        visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

    # sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.num_epochs , batch_num, len(train_loader),
    # mean_discriminator_loss/len(train_loader), mean_generator_content_loss/len(train_loader), 
    # mean_generator_adversarial_loss/len(train_loader), mean_generator_total_loss/len(train_loader)))


        #### Validate the generator ###
    generator.eval()
    out_path = 'training_results/STL/ratio_0.30/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:          ## We probably dont need the val_hr_restore image 
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            
            lr = val_lr
            hr = val_hr

            if torch.cuda.is_available():
                lr = lr.to(device)
                hr = hr.to(device)
            
            sr =generator(lr)
    
            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr']   = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim']   = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to HR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))
    
        #     val_images.extend(
        #         [display_transform()(lr.data.cpu().squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
        #          display_transform()(sr.data.cpu().squeeze(0))])
        
        # val_images = torch.stack(val_images)
        # val_images = torch.chunk(val_images, val_images.size(0) // 15)
        # val_save_bar = tqdm(val_images[0:5], desc='[saving training results]')
        # index = 1

        # for image in val_save_bar:
        #     image = utils.make_grid(image, nrow=3, padding=5)
        #     utils.save_image(image, out_path + extension+'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #     index += 1
        
        cur_psnr = valing_results['psnr']
        cur_ssim = valing_results['ssim']
        
    if cur_ssim>best_ssim and cur_psnr>best_psnr:
        best_ssim = cur_ssim
        best_psnr = cur_psnr
    
        torch.save(generator.state_dict(), model_file_final % opt.out)
        torch.save(discriminator.state_dict(), model_file_discriminator % opt.out)


              ### Store loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

            ### Save the Statistics to a csv File ## 
    if epoch % 1 == 0 and epoch != 0:
        out_path = 'statistics/For_Paper/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(epoch+1))
        data_frame.to_csv(out_path + save_file_final, index_label='Epoch')


