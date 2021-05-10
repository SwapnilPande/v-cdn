import multiprocessing as mp
import os
import time

import gym
import d4rl_pybullet

from PIL import Image

import cv2
import numpy as np
import imageio
import scipy.misc
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset

from utils import rand_float, rand_int
from utils import init_stat, combine_stat, load_data, store_data
from utils import resize, crop
from utils import adjust_brightness, adjust_saturation, adjust_contrast, adjust_hue

DEVICE = "cuda:2"

def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.0
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).to(DEVICE))
            data[i] = (data[i] - s[:, 0]) / s[:, 1]
    else:
        for i in range(len(stat)):
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).to(DEVICE))
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
    return data


class D4RLDataset(Dataset):

    def __init__(self, env_name, args, phase, trans_to_tensor=None):
        self.env_name = env_name

        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor

        self.data_dir = os.path.join(self.args.dataf, phase)

        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.stat = None

        os.system('mkdir -p ' + self.data_dir)

        self.data_names = ['states', 'actions', 'scene_params']

        self.load_data()
        #TODO get validation data working correctly
        ratio = self.args.train_valid_ratio
        n_rollout = len(self.offline_data['rewards'])//self.args.time_step
        if phase in {'train'}:
            self.n_rollout = int(n_rollout * ratio)
        elif phase in {'valid'}:
            self.n_rollout = n_rollout - int(n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.time_step

        self.node_params = self.joint_params = torch.tensor(
            [[0, -2.617 ,  0  ,0,   0,   0,  0.35, 100/120],
            [0,  -.785, .785 ,0, 0.1,   0,  0.1 ,  100/120],
            [0, -2.617 ,  0   ,0,   0,   0,  0.35, 100/120],
            [0,  -.785, .785 ,0, 0.1,   0,  0.1 , 100/120],
            [0, -2.617 ,  0  ,0,  0  ,0, 1.05  , 100/120],
            [0, -2.617 ,  0   ,0,  0  ,0, 1.05  , 100/120]]
        ).float()


    def load_data(self):
        self.env = gym.make(self.env_name)

        self.offline_data = self.env.get_dataset()

        if self.env_name == "halfcheetah-bullet-mixed-v0":
            self.link_params = {
                0: np.array([0.046, .145]),
                1: np.array([0.046, .15]),
                2: np.array([0.046, .094]),
                3: np.array([0.046, .133]),
                4: np.array([0.046, .106]),
                5: np.array([0.046, .07]),
                6: np.array([0.046, 1.0])
            }

        elif self.env_name == "walker2d-bullet-mixed-v0":
            self.link_params = {
                0: np.array([0.05, 0.45]),
                1: np.array([0.04, 0.5 ]),
                2: np.array([0.06, 0.2 ]),
                3: np.array([0.05, 0.45]),
                4: np.array([0.04, 0.5 ]),
                5: np.array([0.06, 0.2]),
                6: np.array([0.05, 0.4])
            }


        # train_size = int(len(self.dataset) * self.train_val_split)
        # val_size = len(self.dataset) - train_size

        # self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))


    def __len__(self):
        length = self.n_rollout * (self.args.time_step - self.args.n_his - self.args.n_roll + 1)
        return length

    def __getitem__(self, idx):
        args = self.args

        # args.time_step - number of time steps per episode
        # Offset is the last index in the episode that we can start at
        offset = args.time_step - args.n_his - args.n_roll + 1
        src_rollout = idx // offset
        src_timestep = idx % offset

        '''
        used for dynamics modeling
        '''

        #TODO validate that this makes sense
        # load images for graph inference
        infer_st_idx = rand_int(0, args.time_step - args.n_identify + 1)

        # if using detected keypoints
        kps_pred = self.offline_data["observations"][(args.time_step)*src_rollout:args.time_step*(src_rollout+1)][::args.frame_offset]

        kps_preload_temp = np.concatenate([
            kps_pred[infer_st_idx : infer_st_idx + args.n_identify],
            kps_pred[src_timestep : src_timestep + args.n_his + args.n_roll]], 0)

        kps_preload = np.zeros(shape = (kps_preload_temp.shape[0], args.n_kp, args.state_dim))

        kps_preload[:, 0,0] = kps_preload_temp[:, 8]
        kps_preload[:, 0,1] = kps_preload_temp[:, 9]


        kps_preload[:, 1,0] = kps_preload_temp[:, 10]
        kps_preload[:, 1,1] = kps_preload_temp[:, 11]

        kps_preload[:, 2,0] = kps_preload_temp[:, 12]
        kps_preload[:, 2,1] = kps_preload_temp[:, 13]

        kps_preload[:, 3,0] = kps_preload_temp[:, 14]
        kps_preload[:, 3,1] = kps_preload_temp[:, 15]

        kps_preload[:, 4,0] = kps_preload_temp[:, 16]
        kps_preload[:, 4,1] = kps_preload_temp[:, 17]

        kps_preload[:, 5,0] = kps_preload_temp[:, 18]
        kps_preload[:, 5,1] = kps_preload_temp[:, 19]

        kps_preload = torch.FloatTensor(kps_preload)


        # get action

        actions_raw = self.offline_data["actions"][(args.time_step)*src_rollout:args.time_step*(src_rollout+1)][::args.frame_offset]


        actions_id = actions_raw[infer_st_idx:infer_st_idx + args.n_identify]
        actions_dy = actions_raw[src_timestep:src_timestep + args.n_his + args.n_roll]

        actions = np.concatenate([actions_id, actions_dy], 0)

        # if using preloaded keypoints
        return kps_preload, actions, self.node_params


