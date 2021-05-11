import os
import time
import random
import itertools
import matplotlib.pyplot as plt
from numpy.lib.financial import _npv_dispatcher

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 12

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import gen_args
from data_d4rl import D4RLDataset
from models_kp import KeyPointNet
from models_dy import DynaNetGNN, HLoss
from utils import count_parameters, Tee, AverageMeter, to_np, to_var, norm, set_seed

from data import normalize, denormalize

from torch.distributions.multivariate_normal import MultivariateNormal

DEVICE = "cuda:0"

args = gen_args()

use_gpu = torch.cuda.is_available()

set_seed(args.random_seed)


# used for cnn encoder, minimum input observation length
min_res = args.min_res


if args.stage == 'dy':

    if args.dy_model == 'mlp':
        model_dy = DynaNetMLP(args, use_gpu=use_gpu)
    elif args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)

    # print model #params
    print("model #params: %d" % count_parameters(model_dy))

    if args.eval_dy_epoch == -1:
        model_dy_path = os.path.join(args.outf_dy, 'net_best_dy.pth')
    else:
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.eval_dy_epoch, args.eval_dy_iter))

    print("Loading saved ckp from %s" % model_dy_path)
    model_dy.load_state_dict(torch.load(model_dy_path, map_location={'cuda:0':DEVICE}))
    model_dy.eval()

if use_gpu:
    model_dy.to(DEVICE)


criterionMSE = nn.MSELoss()
criterionH = HLoss()


'''
data
'''
data_dir = os.path.join(args.dataf, args.eval_set)


data_names = ['states', 'actions']


trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


'''
store results
'''
os.system('mkdir -p ' + args.evalf)

log_path = os.path.join(args.evalf, 'log.txt')
tee = Tee(log_path, 'w')

dataset = D4RLDataset("halfcheetah-bullet-mixed-v0", args, phase='valid', trans_to_tensor=trans_to_tensor)

dataloader = DataLoader(
    dataset, batch_size=args.batch_size,
    shuffle = False,
    num_workers=args.num_workers)

bar = ProgressBar(len(dataloader))

fwd_loss_mse = []

loss_mse = 0.0
for i, data in bar(enumerate(dataloader)):

    if use_gpu:
        if isinstance(data, list):
            # nested transform
            data = [[d.to(DEVICE) for d in dd] if isinstance(dd, list) else dd.to(DEVICE) for dd in data]
        else:
            data = data.to(DEVICE)
    # print()
    # print("Eval # %d / %d" % (roll_idx, ls_rollout_idx[-1]))

    kps, actions, node_params = data

    B = 1
    n_samples = args.n_identify + args.n_his + args.n_roll
    n_kp = 6
    n_identify = args.n_identify
    n_his = args.n_his

    n_identify = 100
    kps = kps.view(B, n_samples, n_kp, args.state_dim)
    kps_id, kps_dy = kps[:, :n_identify], kps[:, n_identify:]

    actions_id, actions_dy = actions[:, :n_identify], actions[:, n_identify:]

    graph = model_dy.graph_inference(kps_id, actions_id, env=args.env, node_params=node_params)

    edge_attr, edge_type_logits = graph[1], graph[3]

    idx_pred = torch.argmax(edge_type_logits, dim=3)


    edges = []
    joints = ['bthigh', 'bshin', 'bfoot', 'fthigh','fshin', 'ffoot', 'torso']
    # for j in range(7):
    #     for k in range(7):
    #         edges.append(idx_pred[0,j,k].item())
    # print(edges)
    #         # print(" {} -> {}, edge {}".format(joints[j],joints[k] , idx_pred[0,j,k]))

    # for j in range(3):
    #     print("{}: {}".format(j, torch.sum(idx_pred == j)))


    eps = args.gauss_std
    kp_cur = kps_dy[:, :n_his].view(B, n_his, n_kp, args.state_dim)
    covar_gt = torch.FloatTensor(torch.eye(args.state_dim) * eps).to(DEVICE)
    # TODO Check this is correct dimensions
    covar_gt = covar_gt.view(1, 1, 1, -1).repeat(B, n_his, n_kp, 1)
    kp_cur = torch.cat([kp_cur, covar_gt], 3)


    for j in range(args.n_roll):
        # kp_desired
        # Retrieve keypoint at next time step
        kp_des = kps_dy[:, n_his + j]

        # predict the feat and hmap at the next time step
        #Retrieve current action
        action_cur = actions_dy[:, j : j + n_his] if actions is not None else None

        if args.dy_model == 'gnn':
            # kp_pred: B x n_kp x 2
            kp_pred = model_dy.dynam_prediction(kp_cur, graph, action_cur, env=args.env, node_params=node_params)
            mean_cur, covar_cur = kp_pred[:, :, :args.state_dim], kp_pred[:, :, args.state_dim:].view(B, n_kp,
                                                                                                      args.state_dim,
                                                                                                      args.state_dim)

            mean_des, covar_des = kp_des, covar_gt[:, 0].view(B, n_kp, args.state_dim, args.state_dim)

            m_cur = MultivariateNormal(mean_cur, scale_tril=covar_cur)
            m_des = MultivariateNormal(mean_des, scale_tril=covar_des)

            loss_mse_cur = criterionMSE(mean_cur, mean_des)
            loss_mse += loss_mse_cur.detach() / args.n_roll

        # update feat_cur and hmap_cur
        kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)



print(loss_mse/len(dataloader))


    # if args.env in ['Ball']:
    #     graph_gt, graph_pred, over_time_results, fwd_loss_mse_cur = evaluate(
    #         roll_idx, video=args.store_demo, image=args.store_demo)
    # elif args.env in ['Cloth']:
    #     gt_pred, over_time_results, fwd_loss_mse_cur = evaluate(
    #         roll_idx, video=args.store_demo, image=args.store_demo)

    # fwd_loss_mse.append(fwd_loss_mse_cur)

    # if args.env in ['Ball']:
    #     edge_acc_over_time_record[roll_idx] = over_time_results[0]
    #     edge_ent_over_time_record[roll_idx] = over_time_results[1]
    #     edge_cor_over_time_raw_record.append(over_time_results[2])
    # elif args.env in ['Cloth']:
    #     edge_ent_over_time_record[roll_idx] = over_time_results

fwd_loss_mse = np.array(fwd_loss_mse)
print()
print('MSE on forward prediction', fwd_loss_mse.shape)
for i in range(fwd_loss_mse.shape[1]):
    print('Step:', i, 'mean: %.6f' % np.mean(fwd_loss_mse[:, i]), 'std: %.6f' % np.std(fwd_loss_mse[:, i]))



def plot_data_mean(ax, data, color, label):
    m, lo, hi = np.mean(data, 0), \
            np.mean(data, 0) - np.std(data, 0), \
            np.mean(data, 0) + np.std(data, 0)
    T = len(m)
    x = np.arange(min_res, min_res + T)
    ax.plot(x, m, '-', color=color, alpha=0.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def plot_data_median(ax, data, color, label):
    m, lo, hi = np.median(data, 0), \
            np.quantile(data, 0.25, 0), \
            np.quantile(data, 0.75, 0)
    T = len(m)
    x = np.arange(min_res, min_res + T)
    ax.plot(x, m, '-', color=color, alpha=0.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


# plot edge accuracy over time
if args.env in ['Ball']:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    plot_data_median(ax, edge_acc_over_time_record, color='b', label='Acc')

    # plt.legend(loc='best', fontsize=12)
    plt.xlabel('# of observation frames', fontsize=15)
    plt.ylabel('Accuracy on edge type', fontsize=15)
    plt.xlim([min_res, args.identify_ed_idx - args.identify_st_idx])
    plt.ylim([0.6, 1])
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(args.evalf, 'acc.png'))
    plt.savefig(os.path.join(args.evalf, 'acc.pdf'))
    plt.show()


# plot edge entropy over time
if args.env in ['Ball']:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    plot_data_median(ax, edge_ent_over_time_record, color='b', label='Entropy')

    # plt.legend(loc='best', fontsize=12)
    plt.xlabel('# of observation frames', fontsize=15)
    plt.ylabel('Entropy on edge type', fontsize=15)
    plt.xlim([min_res, args.identify_ed_idx - args.identify_st_idx])
    plt.ylim([0.23, 0.34])
    plt.yticks(np.arange(0.24, 0.35, 0.02))
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(args.evalf, 'ent.png'))
    plt.savefig(os.path.join(args.evalf, 'ent.pdf'))
    plt.show()


# plot edge attr correlation over time
if args.env in ['Ball']:
    edge_cor_over_time_record = []
    for idx_rel in range(args.edge_st_idx, len(edge_cor_over_time_raw_record[0][0])):
        edge_cor_over_time_cur = np.zeros(
            (args.identify_ed_idx - args.identify_st_idx - min_res + 1))

        for i in range(len(edge_cor_over_time_raw_record[0])):
            edge_attr_gt = []
            edge_attr_pred = []

            for j in range(len(edge_cor_over_time_raw_record)):
                edge_attr_gt.append(edge_cor_over_time_raw_record[j][i][idx_rel][1])
                edge_attr_pred.append(edge_cor_over_time_raw_record[j][i][idx_rel][0])

            edge_attr_gt = np.concatenate(edge_attr_gt).reshape(-1)
            edge_attr_pred = np.concatenate(edge_attr_pred).reshape(-1)
            edge_cor_over_time_cur[i] = np.corrcoef(edge_attr_gt, edge_attr_pred)[0, 1]

        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        # plot_data_median(ax, edge_cor_over_time_record, color='b', label='Cor')
        plt.plot(np.arange(min_res, args.identify_ed_idx - args.identify_st_idx + 1),
                 np.abs(edge_cor_over_time_cur))

        plt.xlabel('# of observation frames', fontsize=15)
        plt.ylabel('Correlation on edge attr (Abs)', fontsize=15)
        plt.xlim([min_res, args.identify_ed_idx - args.identify_st_idx])
        plt.ylim([0.8, 0.95])
        plt.yticks(np.arange(0.8, 1.0, 0.05))
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(args.evalf, 'cor_%d.png' % idx_rel))
        plt.savefig(os.path.join(args.evalf, 'cor_%d.pdf' % idx_rel))
        plt.show()

        edge_cor_over_time_record.append(edge_cor_over_time_cur)

# plot the scatter plot on attr at the last step
if args.env in ['Ball']:
    for idx_rel in range(args.edge_st_idx, len(edge_cor_over_time_raw_record[0][0])):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)

        attr_pred = []
        attr_gt = []
        for i in range(len(edge_cor_over_time_raw_record)):
            attr_pred.append(edge_cor_over_time_raw_record[i][-1][idx_rel][0])
            attr_gt.append(edge_cor_over_time_raw_record[i][-1][idx_rel][1])

        attr_pred = np.concatenate(attr_pred, 0).reshape(-1)
        attr_gt = np.concatenate(attr_gt, 0).reshape(-1)

        if idx_rel == 1:
            idx = np.logical_and(attr_pred < 4.5, attr_gt >= 20)
            attr_gt = attr_gt[idx]
            attr_pred = attr_pred[idx]
        elif idx_rel == 2:
            idx = attr_gt <= 130
            attr_gt = attr_gt[idx]
            attr_pred = attr_pred[idx]

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = \
                stats.linregress(attr_gt, attr_pred)
        # print(slope, intercept, r_value, p_value, std_err)

        plt.scatter(attr_gt, attr_pred, c='r', s=4)
        if idx_rel == 1:
            plt.xticks(np.arange(20, 121, 20))
        elif idx_rel == 2:
            plt.xticks(np.arange(30, 131, 20))
        plt.xlabel('Ground truth hidden confounder')
        plt.ylabel('Predicted edge parameter')
        plt.tight_layout(pad=0.8)
        plt.savefig(os.path.join(args.evalf, 'cor_raw_%d.png' % idx_rel))
        plt.savefig(os.path.join(args.evalf, 'cor_raw_%d.pdf' % idx_rel))
        plt.show()


# store data for plotting
if args.env in ['Ball']:
    # edge_acc_over_time: n_roll x n_timestep

    record_names = ['edge_acc_over_time', 'edge_cor_over_time', 'fwd_loss_mse']
    if args.baseline == 1:
        record_path = os.path.join(args.evalf, 'rec_%d_baseline.h5' % args.n_kp)
    else:
        record_path = os.path.join(args.evalf, 'rec_%d.h5' % args.n_kp)

    store_data(
        record_names,
        [edge_acc_over_time_record, edge_cor_over_time_record, fwd_loss_mse],
        record_path)

    print()
    print('Edge Accuracy')
    print('%.2f%%, std: %.6f' % (
        np.mean(edge_acc_over_time_record[:, -1]) * 100.,
        np.std(edge_acc_over_time_record[:, -1])))

    print()
    print('Correlation on Attributes')
    for i in range(len(edge_cor_over_time_record)):
        print('#%d:' % i, edge_cor_over_time_record[i][-1])

elif args.env in ['Cloth']:

    record_names = ['fwd_loss_mse']
    if args.baseline == 1:
        record_path = os.path.join(args.evalf, 'rec_%d_baseline.h5' % args.n_kp)
    else:
        record_path = os.path.join(args.evalf, 'rec_%d.h5' % args.n_kp)

    store_data(record_names, [fwd_loss_mse], record_path)


