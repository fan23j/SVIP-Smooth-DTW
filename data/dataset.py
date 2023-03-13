import torch
from torch.utils import data
from torchvision import transforms as tf
import logging
import os
import numpy as np
from PIL import Image
import cv2
import random
import pudb
import pdb

from data.label import LABELS

logger = logging.getLogger('Sequence Verification')


class VerificationDataset(data.Dataset):

    def __init__(self,
                 cfg,
                 mode='train',
                 dataset_name='EV',
                 txt_path=None,
                 normalization=None,
                 num_clip=16,
                 augment=True,
                 num_sample=600):

        assert mode in [
            'train', 'test', 'val'], 'Dataset mode is expected to be train, test, or val. But get %s instead.' % mode
        self.mode = mode
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.normalization = normalization
        self.num_clip = num_clip
        self.augment = augment
        if augment:
            self.aug_flip = True
            self.aug_crop = True
            self.aug_color = True
            self.aug_rot = True
        self.num_sample = num_sample  # num of pairs randomly selected from all training pairs
        self.data_list = [line.strip()
                          for line in open(txt_path, 'r').readlines()]
        # sort data_list by label
        self.data_list = sorted(self.data_list, key=lambda x: x.split('/')[0])

        logger.info('Successfully construct dataset with [%s] mode and [%d] samples randomly selected from [%d] samples' % (
            mode, len(self), len(self.data_list)))

    def __getitem__(self, index):
        data_path = self.data_list[index]
        data_path_split = data_path.strip().split(' ')
        sample = {
            # 'index': index,
            'data': data_path,
            'clips1': self.sample_clips(data_path_split[0]),
            'clips2': self.sample_clips(data_path_split[2]),
            'labels1': LABELS[self.dataset_name][self.mode].index(data_path_split[1]) if self.mode == 'train' else data_path_split[1],
            'labels2': LABELS[self.dataset_name][self.mode].index(data_path_split[3]) if self.mode == 'train' else data_path_split[3]
        }

        return sample

    def __len__(self):
        if self.mode == 'train':
            return self.num_sample
        else:
            return len(self.data_list)

    def sample_clips(self, video_dir_path):
        video_dir_path = self.cfg.DATASET.BASE_PATH + video_dir_path
        all_frames = os.listdir(video_dir_path)
        all_frames = [x for x in all_frames if '_' not in x]

        # Evenly divide a video into [self.num_clip] segments
        segments = np.linspace(0, len(all_frames) - 2,
                               self.num_clip + 1, dtype=int)

        sampled_clips = []
        num_sampled_per_segment = 1 if self.mode == 'train' else 3

        # for debug
        idx_list = []
        # for debug

        for i in range(num_sampled_per_segment):
            sampled_frames = []
            for j in range(self.num_clip):
                if self.mode == 'train':
                    frame_index = np.random.randint(
                        segments[j], segments[j + 1])
                else:
                    frame_index = segments[j] + \
                        int((segments[j + 1] - segments[j]) / 4) * (i + 1)
                    idx_list.append(frame_index)  # for debug
                sampled_frames.append(self.sample_one_frame(
                    video_dir_path, frame_index))
            sampled_clips.append(self.preprocess(sampled_frames))

            # for debug
            # for idx, img in enumerate(sampled_frames):
            #    path = str(idx_list[idx]) + '.jpg'
            #    img.save(path)

        return sampled_clips

    def sample_one_frame(self, data_path, frame_index):

        frame_path = os.path.join(
            data_path, 'image-' + str(frame_index + 1) + '.jpeg')
        try:
            frame = cv2.imread(frame_path)
            # Convert RGB to BGR and transform to PIL.Image
            frame = Image.fromarray(frame[:, :, [2, 1, 0]])
            return frame
        except:
            logger.info('Wrong image path %s' % frame_path)
            exit(-1)

    def preprocess(self, frames, apply_normalization=True):
        # Apply augmentation and normalization on a clip of frames

        # Data augmentation on the frames
        transforms = []
        if self.augment:
            # Flip
            if np.random.random() > 0.5 and self.aug_flip:
                transforms.append(tf.RandomHorizontalFlip(1))

            # Random crop
            if np.random.random() > 0.5 and self.aug_crop:
                transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))

            # Color augmentation
            if np.random.random() > 0.5 and self.aug_color:
                transforms.append(tf.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

            # # Rotation
            # if np.random.random() > 0.5 and self.aug_rot:
            #     transforms.append(tf.RandomRotation(30))

        # PIL image to tensor
        transforms.append(tf.ToTensor())

        # Normalization
        if self.normalization is not None and apply_normalization:
            transforms.append(tf.Normalize(
                self.normalization[0], self.normalization[1]))

        transforms = tf.Compose(transforms)

        frames = torch.cat([transforms(frame).unsqueeze(-1)
                           for frame in frames], dim=-1)

        return frames


class RandomSampler(data.Sampler):
    # randomly sample [len(self.dataset)] items from [len(self.data_list))] items

    def __init__(self, dataset, txt_path, shuffle=False, mode="train"):
        self.dataset = dataset
        self.data_list = [line.strip()
                          for line in open(txt_path, 'r').readlines()]
        self.shuffle = shuffle
        self.mode = mode

    def __iter__(self):
        tmp = []

        # if (self.mode == "train"):
        #     # sample random item from data_list
        #     random_item = random.choice(self.data_list)
        #     # retrieve label from random_item and filter data_list based on that label
        #     random_label = random_item.strip().split(' ')[1]
        #     print(random_label)
        #     filtered_list = [self.data_list.index(item) for item in self.data_list if item.strip().split(' ')[
        #         1] == random_label]
        #     # shuffle list
        #     random.shuffle(filtered_list)
        #     # truncate list to closest multiple of batch_size (8) less than len(filtered_list)
        #     truncated_len = len(filtered_list) - (len(filtered_list) % 8)
        #     while len(filtered_list) > truncated_len:
        #         filtered_list.pop()
        #     tmp = filtered_list
        # else:
        tmp = random.sample(range(len(self.data_list)), len(self.dataset))
        if not self.shuffle:
            tmp.sort()

        # pdb.set_trace()
        return iter(tmp)

    def __len__(self):
        return len(self.dataset)


def load_dataset(cfg):

    ImageNet_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    dataset = VerificationDataset(
        cfg=cfg,
        mode=cfg.DATASET.MODE,
        dataset_name=cfg.DATASET.NAME,
        txt_path=cfg.DATASET.TXT_PATH,
        normalization=ImageNet_normalization,
        num_clip=cfg.DATASET.NUM_CLIP,
        augment=cfg.DATASET.AUGMENT,
        num_sample=cfg.DATASET.NUM_SAMPLE)

    sampler = RandomSampler(dataset, cfg.DATASET.TXT_PATH,
                            cfg.DATASET.SHUFFLE, cfg.DATASET.MODE)

    loaders = data.DataLoader(dataset=dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=False,
                              sampler=sampler,
                              drop_last=False,
                              num_workers=cfg.DATASET.NUM_WORKERS,
                              pin_memory=True)

    return loaders


if __name__ == "__main__":

    import sys
    sys.path.append(
        '/root/workspace/SVIP')
    from configs.defaults import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/train_resnet_config.yml')

    train_loader = load_dataset(cfg)

    for iter, sample in enumerate(train_loader):
        print(sample.keys())
        print(sample['clips1'][0].size())
        print(sample['labels1'])
        break
