import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import copy
import torch
import torchvision.transforms as transforms
import scipy.io as scio
import numpy as np

rng = np.random.RandomState(2020)

def test_np_load_frame(filename , resize_h , resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized

def np_load_frame(filename, resize_height, resize_width, grayscale=False):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray [256 , 256 , 3(if RGB else 1 for gray)]
    """
    if grayscale:
        image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

class Reconstruction3DDataLoader(data.Dataset):
    def __init__(
            self,
            video_folder,
            transform,
            resize_height,
            resize_width,
            grayscale,
            num_frames=9,                   # ############################################################此处更改
            img_extension='.jpg',
            dataset='other',
            jump=[2],
            hold=[2],
            return_normal_seq=False,

    ):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames
        self.extension = img_extension
        self.dataset = dataset
        self.jump = jump
        self.hold = hold
        self.return_normal_seq = return_normal_seq  # for fast and slow moving
        self.grayscale = grayscale

        self.setup()

        self.samples, self.background_models = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*/'))

        for video in sorted(videos):
            video_name = video.split('/')[-2] # the number of video from 1 to n
            #print(video_name)
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video , '*' + self.extension))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            #print(self.videos[video_name]['length'])

    def get_all_samples(self):
        frames = []
        background_models = []
        videos = glob.glob(os.path.join(self.dir, '*/'))

        for video in sorted(videos):
            video_name = video.split('/')[-2] # the number of video from 1 to n

            for i in range(self.videos[video_name]['length'] - self._num_frames + 1):
                frames.append(self.videos[video_name]['frame'][i])
                #print(self.videos[video_name]['frame'][i])

        return frames, background_models

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(0 , self._num_frames):   # ############################################################此处更改
            image = np_load_frame(
                self.videos[video_name]['frame'][frame_name + i],
                self._resize_height,
                self._resize_width,
                grayscale=self.grayscale
            )
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.stack(batch, axis=1)

    def __len__(self):
        return len(self.samples)

class test_dataset():
    def __init__(self , video_folder):
        self.img_h = 256
        self.img_w = 256
        self.clip_length = 9   # ############################################################此处更改
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs) - (self.clip_length) + 1

    def __getitem__(self, indice):
        video_clips = []
        for frame_id in range(indice , indice + self.clip_length):   # ############################################################此处更改
            video_clips.append(test_np_load_frame(self.imgs[frame_id] , self.img_h , self.img_w))

        video_clips = np.array(video_clips)
        video_clips = torch.from_numpy(video_clips).transpose(0 , 1)
        return video_clips

class Label_loader:
    def __init__(self , data_path , video_folders , mat_path , dataset_type='ped2' ,):
        self.name = dataset_type
        self.frame_path = data_path
        self.mat_path = f'{mat_path}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)

            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]

                sub_video_gt[start: end] = 1

            all_gt.append(sub_video_gt)

        return all_gt


if __name__ == '__main__':


    traindata = Reconstruction3DDataLoader(
        '/media/ttkx/university/data/other/training' , transforms.Compose([transforms.ToTensor()]) , 256 , 256 , num_frames=9 , grayscale=False
    )
    train_batch = data.DataLoader(
        traindata , batch_size=4 , shuffle=False , num_workers=0
    )
    for i in train_batch:
        print(i.shape)
