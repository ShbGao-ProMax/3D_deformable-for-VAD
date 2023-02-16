import cv2
import time
import datetime
import torch
from torch.utils.data import DataLoader
import argparse
import random
import dataset
import os
from models.pix2pix_network import PixelDiscriminator
from models.unet import UNet
from utils import *
from losses import *
import torchvision.transforms as transforms
from evaluate import val
from tensorboardX import SummaryWriter

# - - - - - - - - folder create - - - - - - -
if not os.path.exists("runs"):
    os.mkdir("runs")
if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("Visualization"):
    os.mkdir("Visualization")
if not os.path.exists("weights"):
    os.mkdir("weights")
    os.mkdir("weights/early_stop")

writer1 = SummaryWriter()

train_data_path = '/lustre/home/shbgao/data/other_ped2/training'
test_data_path = '/lustre/home/shbgao/data/other_ped2/testing'
dataset_type = 'other_ped2'
mat_path = '/lustre/home/shbgao/data/other_ped2'

parser = argparse.ArgumentParser(description='autoencoder-3D')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--dataset', default='other_ped2', type=str, help='The name of the dataset to train.')
parser.add_argument('--resume', default='', type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--epoch' , default=100 , type=int , help='epoch num for trainning')
parser.add_argument('--g_lr' , default=0.0006 , type=float)
parser.add_argument('--d_lr' , default=0.00006 , type=float)
args = parser.parse_args()

generator = UNet(3 , 3).cuda()
discriminator = PixelDiscriminator(input_nc=3).cuda()
optimizer_G = torch.optim.Adam(generator.parameters() , lr = args.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters() , lr = args.d_lr)

if args.resume:
    generator.load_state_dict(torch.load(args.resume)['net_g'])
    discriminator.load_state_dict(torch.load(args.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(args.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(args.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss_cuda(channels=3).cuda()
intensity_loss = Intensity_Loss().cuda()

traindataset = dataset.Reconstruction3DDataLoader(
    video_folder = train_data_path,
    transform = transforms.Compose([transforms.ToTensor()]),
    resize_width = 256,
    resize_height = 256,
    grayscale = False,
    num_frames=9,
)

traindataloader = DataLoader(
    dataset = traindataset,
    batch_size = args.batch_size,
    shuffle = False,
    num_workers = args.num_workers,
    drop_last = True,
)

generator = generator.train()
discriminator = discriminator.train()
epoch=0

try:
    while epoch < args.epoch:
        epoch_psnr = 0
        show_num = 0
        for indice, data in enumerate(traindataloader):

            total_batch = len(traindataloader)

            input_frames = data[:, :, 0:8, :, :].cuda()
            target_frames = data[:, :, 1:9, :, :].cuda()

            # print(input_frames.shape)
            G_frame = generator(input_frames)
            # print(G_frame.shape)

            G_total_loss = 0
            D_total_loss = 0
            frame_number = args.batch_size * G_frame.shape[2]

            for i in range(args.batch_size):

                for j in range(G_frame.shape[2]):
                    G_in, target_in = G_frame[i, :, j, :, :], target_frames[i, :, j, :, :]  # [3 , 256 , 256] , [3 , 256 , 256]
                    _G_in, _target_in = G_in.unsqueeze(0), target_in.unsqueeze(0)  # [1 , 3 , 256 , 256] , [1 , 3 , 256 , 256]

                    inte_l = intensity_loss(G_in, target_in)
                    grad_l = gradient_loss(_G_in, _target_in)
                    g_l = adversarial_loss(discriminator(_G_in))

                    G_l_t = (1. * inte_l + 1. * grad_l + 0.05 * g_l)
                    D_l_t = (discriminate_loss(discriminator(_target_in), discriminator(_G_in.detach())))

                    G_total_loss += G_l_t
                    D_total_loss += D_l_t
#  - - - - - - - step - - - - - - -

            optimizer_D.zero_grad()
            D_l_t.backward()
            optimizer_G.zero_grad()
            G_l_t.backward()
            optimizer_D.step()
            optimizer_G.step()

            torch.cuda.synchronize()

            if indice % 20 == 0:
                psnr = psnr_error_cuda(_G_in, _target_in)
                epoch_psnr += psnr
                show_num += 1

                print(f"[epoch {epoch} / {args.epoch} ] || [item {indice} / {total_batch}]")
                print(f"[psnr : {psnr:.5f}] || [G_loss_total : {G_l_t:.5f}] || [D_loss_total : {D_l_t:.5f}]")

            if indice % 50 == 0:
                save_pic = stack_Visualization(target_in, G_in)
                cv2.imwrite(f'Visualization/epoch{epoch}_indice{indice}.jpg', save_pic)

        if epoch % 1 == 0 and epoch != 0 and epoch>5:
            model_dict = {
                'net_g': generator.state_dict(),
                'optimizer_g': optimizer_G.state_dict(),
                'net_d': discriminator.state_dict(),
                'optimizer_d': optimizer_D.state_dict()
            }
            torch.save(
                model_dict, f'weights/epoch{epoch}_folder{indice}.pth'
            )
            print('model saved!')
            print(f'weights/epoch{epoch}.pth')

        print("- - - - - - - - - - -")
        print(f"[epoch_average_psnr : {epoch_psnr / show_num}] || [statistics_num : {show_num}]")
        print("- - - - - - - - - - -")

        epoch += 1

        auc = val(
            test_data_path=test_data_path,
            dataset_type=dataset_type,
            mat_path=mat_path,
            model=generator
        )
        #print(auc)

        writer1.add_scalar("auc" , auc , global_step = epoch)
        writer1.add_scalar("psnr" , epoch_psnr / show_num , global_step = epoch)
        generator.train()

except KeyboardInterrupt:

    print(f'stop early! model saved')
    model_dict = {
        'net_g': generator.state_dict(),
        'optimizer_g': optimizer_G.state_dict(),
        'net_d': discriminator.state_dict(),
        'optimizer_d': optimizer_D.state_dict()
    }
    torch.save(model_dict, f'weights/early_stop/epoch{epoch}_early.pth')
