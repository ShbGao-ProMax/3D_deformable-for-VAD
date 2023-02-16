import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Intensity_Loss(nn.Module):
    def __init__(self):
        super(Intensity_Loss, self).__init__()

    def forward(self , gen_frames , gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))

class Gradient_Loss(nn.Module):
    def __init__(self , channels):
        super(Gradient_Loss, self).__init__()

        pos = torch.from_numpy(np.identity(channels , dtype=np.float32)) # 创建一个对角线为1其余为0的方阵
        neg = -1 * pos

        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1)
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1)

    def forward(self , gen_frames , gt_frames):

        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)

class Gradient_Loss_cuda(nn.Module):
    def __init__(self , channels):
        super(Gradient_Loss_cuda, self).__init__()

        pos = torch.from_numpy(np.identity(channels , dtype=np.float32)) # 创建一个对角线为1其余为0的方阵
        neg = -1 * pos

        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self , gen_frames , gt_frames):

        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)

class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss, self).__init__()

    def forward(self , fake_outputs):
        return torch.mean((fake_outputs - 1) ** 2 / 2)

class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss, self).__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)


if __name__ == '__main__':
    pos = torch.from_numpy(np.identity(3, dtype=np.float32))  # 创建一个对角线为1其余为0的方阵
    neg = -1 * pos
    print(pos.shape , neg.shape)
    filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3 , 2 , 0 , 1)
    print(filter_x , filter_x.shape)
    filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1)
    print(filter_y , filter_y.shape)