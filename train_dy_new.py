import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal

from config import gen_args
from data import PhysicsDataset, load_data
from data_d4rl import D4RLDataset
from models_kp import KeyPointNet
from models_dy import DynaNetGNN, HLoss
from utils import rand_int, count_parameters, Tee, AverageMeter, get_lr, to_np, set_seed

args = gen_args()
set_seed(args.random_seed)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

os.system('mkdir -p ' + args.outf_kp)
os.system('mkdir -p ' + args.dataf)

if args.stage == 'dy':
    os.system('mkdir -p ' + args.outf_dy)
    tee = Tee(os.path.join(args.outf_dy, 'train.log'), 'w')
else:
    raise AssertionError("Unsupported env %s" % args.stage)

print(args)

# generate data
trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


## Loading datasets here, switch to our dataset

if(args.env == "half-cheetah"):
    dataset_name = "halfcheetah-bullet-mixed-v0"
elif(args.env == "walker"):
    dataset_name = "walker2d-bullet-mixed-v0"

print(dataset_name)
datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = D4RLDataset(dataset_name, args, phase=phase, trans_to_tensor=trans_to_tensor)

    # if args.gen_data:
    #     datasets[phase].gen_data()
    # else:
    #     datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])

args.stat = datasets['train'].stat

use_gpu = torch.cuda.is_available()


'''
define model for dynamics prediction
'''
if args.stage == 'dy':

    if args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)
    else:
        raise AssertionError("Unknown dy_model %s" % args.dy_model)

    print("model_dy #params: %d" % count_parameters(model_dy))

    if args.dy_epoch >= 0:
        # if resume from a pretrained checkpoint
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.dy_epoch, args.dy_iter))
        print("Loading saved ckp for dynamics net from %s" % model_dy_path)
        model_dy.load_state_dict(torch.load(model_dy_path))

else:
    raise AssertionError("Unknown stage %s" % args.stage)


# criterion
criterionMSE = nn.MSELoss()
criterionH = HLoss()

# optimizer
if args.stage == 'dy':
    params = model_dy.parameters()
else:
    raise AssertionError('Unknown stage %s' % args.stage)

optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)

if use_gpu:
    criterionMSE = criterionMSE.cuda()

    if args.stage == 'dy':
        model_dy = model_dy.cuda()
    else:
        raise AssertionError("Unknown stage %s" % args.stage)


if args.stage == 'dy':
    st_epoch = args.dy_epoch if args.dy_epoch > 0 else 0
    log_fout = open(os.path.join(args.outf_dy, 'log_st_epoch_%d.txt' % st_epoch), 'w')
else:
    raise AssertionError("Unknown stage %s" % args.stage)


best_valid_loss = np.inf

# Training loop
for epoch in range(st_epoch, args.n_epoch):
    phases = ['train', 'valid'] if args.eval == 0 else ['valid']

    for phase in phases:


        meter_loss = AverageMeter()
        meter_loss_contras = AverageMeter()

        if args.stage == 'dy':
            model_dy.train(phase == 'train')
            meter_loss_rmse = AverageMeter()
            meter_loss_kp = AverageMeter()
            meter_loss_H = AverageMeter()
            meter_acc = AverageMeter()
            meter_cor = AverageMeter()
            meter_num_edge_per_type = np.zeros(args.edge_type_num)

        bar = ProgressBar(maxval=data_n_batches[phase])
        loader = dataloaders[phase]

        # Loop over dataset
        for i, data in bar(enumerate(loader)):


            if use_gpu:
                if isinstance(data, list):
                    # nested transform
                    data = [[d.cuda() for d in dd] if isinstance(dd, list) else dd.cuda() for dd in data]
                else:
                    data = data.cuda()

            with torch.set_grad_enabled(phase == 'train'):
                if args.stage == 'dy':
                    '''
                    hyperparameter on the length of data
                    '''
                    n_his, n_kp = args.n_his, args.n_kp

                    n_samples = args.n_identify + args.n_his + args.n_roll
                    n_identify = args.n_identify

                    '''
                    load data
                    '''
                    kps_preload, actions, node_params = data
                    kps = kps_preload

                    #TODO Make sure that this is the right dimensions
                    # B = batch_size
                    B = actions.size(0)

                    # elif args.env in ['Cloth']:
                    #     if args.preload_kp == 1:
                    #         # if using preloaded keypoints
                    #         kps_preload, actions = data
                    #     else:
                    #         imgs, actions = data
                    #     kps_gt = None
                    #     B = actions.size(0)

                    #     print(kps_preload.shape)
                    #     exit()

                    '''
                    get detected keypoints -- kps
                    '''
                    # kps: B x (n_identify + n_his + n_roll) x n_kp x 2

                    # Permute the keypoints to make sure the calculation of
                    # edge accuracy is correct.
                    #TODO Do we need this
                    # if i == 0:
                    #     permu_node_idx = np.arange(args.n_kp)

                    #     if args.env in ['Ball']:
                    #         permu_node_list = list(itertools.permutations(np.arange(args.n_kp)))
                    #         import ipdb
                    #         ipdb.set_trace()

                    #         permu_node_error = np.inf
                    #         permu_node_idx = None
                    #         for ii in permu_node_list:
                    #             p = np.array(ii)
                    #             kps_permuted = kps[:, :, p]

                    #             error = torch.mean((kps_permuted - kps_gt)**2).item()
                    #             if error < permu_node_error:
                    #                 permu_node_error = error
                    #                 permu_node_idx = p

                    #         # permu_node_idx = np.array([2, 1, 0, 4, 3])
                    #         print()
                    #         print('Selected node permutation', permu_node_idx)

                    # kps = kps[:, :, permu_node_idx]

                    # (Batch_size x n_samples x n_keypoints x n_state_features)
                    kps = kps.view(B, n_samples, n_kp, args.state_dim)
                    kps_id, kps_dy = kps[:, :n_identify], kps[:, n_identify:]

                    # only train dynamics module
                    kps = kps.detach()

                    actions_id, actions_dy = actions[:, :n_identify], actions[:, n_identify:]

                    '''
                    step #1: identify the dynamics graph
                    '''
                    # randomize the observation length
                    observe_length = rand_int(args.min_res, n_identify + 1)

                    if args.baseline == 1:
                        graph = model_dy.init_graph(
                            kps_id[:, :observe_length], use_gpu=True, hard=True)
                    else:

                        #TODO Make sure that the actions are correctly encoded in the edges
                        graph = model_dy.graph_inference(
                            kps_id[:, :observe_length], actions_id[:, :observe_length],
                            env=args.env, node_params = node_params)

                    # calculate edge calculation accuracy
                    # edge_attr: B x n_kp x n_kp x edge_attr_dim
                    # edge_type_logits: B x n_kp x n_kp x edge_type_num
                    edge_attr, edge_type_logits = graph[1], graph[3]

                    idx_pred = torch.argmax(edge_type_logits, dim=3)
                    idx_pred = idx_pred.data.cpu().numpy()


                    # record the number of edges that belongs to a specific type
                    num_edge_per_type = np.zeros(args.edge_type_num)
                    for tt in range(args.edge_type_num):
                        num_edge_per_type[tt] = np.sum(idx_pred == tt)
                    meter_num_edge_per_type += num_edge_per_type


                    # step #2: dynamics prediction
                    eps = args.gauss_std
                    kp_cur = kps_dy[:, :n_his].view(B, n_his, n_kp, args.state_dim)
                    covar_gt = torch.FloatTensor(torch.eye(args.state_dim) * eps).cuda()
                    # TODO Check this is correct dimensions
                    covar_gt = covar_gt.view(1, 1, 1, -1).repeat(B, n_his, n_kp, 1)
                    kp_cur = torch.cat([kp_cur, covar_gt], 3)

                    loss_kp = 0.
                    loss_mse = 0.

                    edge_type_logits = graph[3].view(-1, args.edge_type_num)

                    loss_H = -criterionH(edge_type_logits, args.prior)

                    for j in range(args.n_roll):

                        # kp_desired
                        # Retrieve keypoint at next time step
                        kp_des = kps_dy[:, n_his + j]

                        # predict the feat and hmap at the next time step
                        #Retrieve current action
                        action_cur = actions_dy[:, j : j + n_his] if actions is not None else None

                        if args.dy_model == 'gnn':
                            # kp_pred: B x n_kp x 2
                            kp_pred = model_dy.dynam_prediction(kp_cur, graph, action_cur, env=args.env, node_params = node_params)
                            mean_cur, covar_cur = kp_pred[:, :, :args.state_dim], kp_pred[:, :, args.state_dim:].view(B, n_kp, args.state_dim, args.state_dim)

                            mean_des, covar_des = kp_des, covar_gt[:, 0].view(B, n_kp, args.state_dim, args.state_dim)

                            m_cur = MultivariateNormal(mean_cur, scale_tril=covar_cur)
                            m_des = MultivariateNormal(mean_des, scale_tril=covar_des)

                            log_prob = (m_cur.log_prob(kp_des) - m_des.log_prob(kp_des)).mean()
                            # log_prob = m_cur.log_prob(kp_des).mean()

                            loss_kp_cur = -log_prob * args.lam_kp
                            # loss_kp_cur = criterionMSE(mean_cur, mean_des) * args.lam_kp
                            # print(criterionMSE(mean_cur, mean_des) * args.lam_kp)
                            loss_kp += loss_kp_cur / args.n_roll

                            loss_mse_cur = criterionMSE(mean_cur, mean_des)
                            loss_mse += loss_mse_cur / args.n_roll

                        # update feat_cur and hmap_cur
                        kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

                    # summarize the losses
                    loss = loss_kp + loss_H

                    # update meter
                    meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
                    meter_loss_kp.update(loss_kp.item(), B)
                    meter_loss_H.update(loss_H.item(), B)
                    meter_loss.update(loss.item(), B)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % args.log_per_iter == 0:
                log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                    phase, epoch, args.n_epoch, i, data_n_batches[phase],
                    get_lr(optimizer))

                if args.stage == 'dy':
                    log += ', kp: %.6f (%.6f), H: %.6f (%.6f)' % (
                        loss_kp.item(), meter_loss_kp.avg,
                        loss_H.item(), meter_loss_H.avg)

                    log += ' [%d' % num_edge_per_type[0]
                    for tt in range(1, args.edge_type_num):
                        log += ', %d' % num_edge_per_type[tt]
                    log += ']'

                    log += ', rmse: %.6f (%.6f)' % (
                        np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                    if args.env in ['Ball']:
                        log += ', acc: %.4f (%.4f)' % (
                            permu_edge_acc, meter_acc.avg)
                        log += ' [%d' % permu_edge_idx[0]
                        for ii in permu_edge_idx[1:]:
                            log += ' %d' % ii
                        log += '], cor: %.4f (%.4f)' % (permu_edge_cor, meter_cor.avg)

                print()
                print(log)
                log_fout.write(log + '\n')
                log_fout.flush()

            if phase == 'train' and i % args.ckp_per_iter == 0:
                if args.stage == 'dy':
                    torch.save(model_dy.state_dict(), '%s/net_dy_epoch_%d_iter_%d.pth' % (args.outf_dy, epoch, i))

        log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
        log += ', [%d' % meter_num_edge_per_type[0]
        for tt in range(1, args.edge_type_num):
            log += ', %d' % meter_num_edge_per_type[tt]
        log += ']'
        print(log)
        log_fout.write(log + '\n')
        log_fout.flush()

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg

                if args.stage == 'dy':
                    torch.save(model_dy.state_dict(), '%s/net_best_dy.pth' % (args.outf_dy))


log_fout.close()
