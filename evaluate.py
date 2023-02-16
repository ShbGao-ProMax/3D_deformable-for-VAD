import argparse
import numpy as np
import os
import torch
from models.unet import UNet
from torch.utils.data import DataLoader
from sklearn import metrics
from dataset import Reconstruction3DDataLoader , test_dataset , Label_loader
from utils import psnr_error_cuda , Visualization ,stack_Visualization
import torchvision.transforms as transforms
import cv2


parser = argparse.ArgumentParser(description='unet-3D')
parser.add_argument('--trained_model', default='epoch0_early.pth', type=str, help='The pre-trained model to evaluate.')

args = parser.parse_args()

def val(test_data_path , dataset_type , mat_path , model = None):

    if model:   # model is for testing during training
        generator = model
        generator.eval()
    else:
        generator = UNet(in_ch=3 , out_ch=3).cuda().eval()
        generator.load_state_dict(torch.load('weights/' + args.trained_model)['net_g'])

    video_folders = os.listdir(test_data_path)
    video_folders.sort()
    video_folders = [os.path.join(test_data_path, aa) for aa in video_folders]

    psnr_group = []

    with torch.no_grad():

        for i , folder in enumerate(video_folders):
            testdataset = test_dataset(folder)
            psnrs = []

            for j , clip in enumerate(testdataset):
                input = clip[: , 0 : 8 , : , :] # torch.Size([3, 8, 256, 256])
                target = clip[: , 8 , : , :].cuda() # torch.Size([3, 256, 256])

                input_frames = input.unsqueeze(0).cuda() # torch.Size([1, 3, 8, 256, 256])

                G_frames = generator(input_frames)

                test_frame = G_frames[: , : , 7 , : , :] # torch.Size([1, 3, 256, 256])

                test_psnr = psnr_error_cuda(test_frame , target.unsqueeze(0)).cpu().detach().numpy()
                psnrs.append(float(test_psnr))

                save_pic = stack_Visualization(target , test_frame.squeeze(0))
                cv2.imwrite(f'results/folder{i}_item{j}.jpg', save_pic)

                torch.cuda.synchronize()

            psnr_group.append(np.array(psnrs))
            #print(f'folder:{i + 1} is done')

        print('\nAll frames were detected, begin to compute AUC.')

        gt_loader = Label_loader(
            data_path = test_data_path,
            video_folders = video_folders,
            mat_path = mat_path,
            dataset_type = dataset_type
        )
        gt = gt_loader()
        assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'
        #for k in range(len(psnr_group)):
            #print(len(psnr_group[k]) , len(gt[k]))

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)

        for i in range(len(psnr_group)):
            distance = psnr_group[i]
            distance -= min(distance)
            distance /= max(distance)  # distance = (distance - min) / (max - min)

            scores = np.concatenate((scores, distance), axis=0)
            labels = np.concatenate((labels, gt[i][8:]), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        print(f'AUC: {auc}\n')
        return auc