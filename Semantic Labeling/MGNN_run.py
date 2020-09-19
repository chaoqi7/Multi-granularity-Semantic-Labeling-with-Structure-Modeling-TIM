from __future__ import division
from __future__ import print_function
from builtins import range

import time
import random
import numpy as np
import json
import os
import sys
import math
import argparse
import ast
from tqdm import tqdm
import logging
from collections import defaultdict
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import torchnet as tnt

import spg
import graphnet
import pointnet
import fine_grained_module
import metrics
from timeit import default_timer as timer
import tanker_dataset


def main():
    parser = argparse.ArgumentParser(description='-granularity Semantic Labeling of Point Clouds for the Measurement of the Rail Tanker Component with Structure Modeling')
    # Tanker Dataset
    parser.add_argument('--odir', default='results', help='Directory to store results')
    parser.add_argument('--resume', default='', help='/home/data2/qc/large_scalepcss/learning/bfSubsequentfine-tuning/results/model.pth.tar')
    parser.add_argument('--db_train_name', default='train')
    parser.add_argument('--db_test_name', default='val')
    parser.add_argument('--TANKER_PATH', default='/home/data2/qc/large_scalepcss/learning/datasets/tanker')
    # Fine seg
    parser.add_argument('--fine_seg_epo', default=1, help='start epoch for seg')
    parser.add_argument('--fine_seg_point_num', default=20000, help='point number for seg')
    # Optimization arguments
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=60, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_multisamp_n', default=1, type=int, help='Average logits obtained over runs with different seeds')
    parser.add_argument('--gpu', type=str, default='3', help='GPU to use [default: GPU 0]')

    # Model
    parser.add_argument('--model_config', default='gru_10,f_6', help='Defines the model as a sequence of layers, see graphnet.py for definitions of respective layers and acceptable arguments. In short: rectype_repeats_mv_layernorm_ingate_concat, with rectype the type of recurrent unit [gru/crf/lstm], repeats the number of message passing iterations, mv (default True) the use of matrix-vector (mv) instead vector-vector (vv) edge filters, layernorm (default True) the use of layernorms in the recurrent units, ingate (default True) the use of input gating, concat (default True) the use of state concatenation')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--edge_attribs', default='delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d', help='Edge attribute definition, see spg_edge_features() in spg.py for definitions.')

    # Point cloud processing
    parser.add_argument('--pc_attribs', default='', help='Point attributes fed to PointNets, if empty then all possible.')
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', default=1, type=int, help='Training augmentation: Bool, Gaussian jittering of all attributes')
    parser.add_argument('--pc_xyznormalize', default=1, type=int, help='Bool, normalize xyz into unit ball, i.e. in [-0.5,0.5]')

    # Filter generating network
    parser.add_argument('--fnet_widths', default='[32,128,64]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--fnet_bnidx', default=2, type=int, help='Layer index to insert batchnorm to. -1=do not insert.')
    parser.add_argument('--edge_mem_limit', default=30000, type=int, help='Number of edges to process in parallel during computation, a low number can reduce memory peaks.')

    # Superpoint graph
    parser.add_argument('--spg_attribs01', default=1, type=int, help='Bool, normalize edge features to 0 mean 1 deviation')
    parser.add_argument('--spg_augm_nneigh', default=100, type=int, help='Number of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_order', default=3, type=int, help='Order of neighborhoods to sample in SPG')
    parser.add_argument('--spg_augm_hardcutoff', default=512, type=int, help='Maximum number of superpoints larger than args.ptn_minpts to sample in SPG')
    parser.add_argument('--spg_superedge_cutoff', default=-1, type=float, help='Artificially constrained maximum length of superedge, -1=do not constrain')

    # Point net used for CoarseModule_pn and FineModule
    parser.add_argument('--ptn_minpts', default=40, type=int, help='Minimum number of points in a superpoint for computing its embedding.')
    parser.add_argument('--ptn_npts', default=128, type=int, help='Number of input points for PointNet.')
    parser.add_argument('--ptn_widths', default='[[64,64,128,128,256], [256,64,32]]', help='PointNet widths')
    parser.add_argument('--ptn_widths_stn', default='[[64,64,128], [128,64]]', help='PointNet\'s Transformer widths')
    parser.add_argument('--ptn_widths_seg', default='[[64,64,128,128,256], [256,64,32]]', help='PointNet widths')
    parser.add_argument('--ptn_nfeat_stn', default=13, type=int, help='PointNet\'s Transformer number of input features')
    parser.add_argument('--ptn_prelast_do', default=0, type=float)
    parser.add_argument('--ptn_mem_monger', default=1, type=int, help='Bool, save GPU memory by recomputing PointNets in back propagation.')

    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    args.ptn_widths = ast.literal_eval(args.ptn_widths)
    args.ptn_widths_stn = ast.literal_eval(args.ptn_widths_stn)
    args.ptn_widths_seg = ast.literal_eval(args.ptn_widths_seg)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)

    set_seed(args.seed, args.cuda)
    logging.getLogger().setLevel(logging.INFO)  #set to logging.DEBUG to allow for more prints

    dbinfo = tanker_dataset.get_info(args)
    create_dataset = tanker_dataset.get_datasets

    # Create model and optimizer
    if args.resume != '':
        if args.resume=='RESUME': args.resume = args.odir + '/model.pth.tar'
        model, optimizer, stats = resume(args, dbinfo)
    else:
        model = create_model(args, dbinfo)
        optimizer = create_optimizer(args, model)
        stats = []

    train_dataset, test_dataset = create_dataset(args)
    ptnCloudEmbedder = pointnet.CloudEmbedder(args)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_decay, last_epoch=args.start_epoch-1)
   
    ############
    def train(epoch):
        """ Trains for one epoch """
        model.train()
    
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=spg.eccpc_collate, num_workers=args.nworkers, shuffle=True, drop_last=True)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
    
        loss_meter = tnt.meter.AverageValueMeter() #coresponding to Coarse-grained Module
        loss2_meter = tnt.meter.AverageValueMeter()#coresponding to Fine-grained Module
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        acc_meter2 = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(5)
        t0 = time.time()

        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            t_loader = 1000*(time.time()-t0)
            model.ecc.set_info(GIs, args.cuda)# put into edge information
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1)
            if args.cuda:
                label_mode, label_vec, segm_size = label_mode_cpu.cuda(), label_vec_cpu.float().cuda(), segm_size_cpu.float().cuda()
    
            else:
                label_mode, label_vec, segm_size = label_mode_cpu, label_vec_cpu.float(), segm_size_cpu.float()
    
            optimizer.zero_grad()
            t0 = time.time()
    
            embeddings = ptnCloudEmbedder.run(model, *clouds_data[0:4])
            outputs = model.ecc(embeddings)
            loss = nn.functional.cross_entropy(outputs, Variable(label_mode))

            """ Node determination """
            mixed_node=np.where(np.argmax(np.array(outputs.data),axis=1)==5)# mixed node
            Nomixed_node=np.where(np.argmax(np.array(outputs.data),axis=1)!=5)
            
            if epoch>args.fine_seg_epo and len(mixed_node[0])>0:
                loss = nn.functional.cross_entropy(outputs[Nomixed_node[0],:], Variable(label_mode[[Nomixed_node[0]]]))
                Sum_slice=[]
                Sum_slice=np.hstack((Sum_slice,mixed_node[0]))
                
                for i in range(len(Sum_slice)):
                    fname=args.TANKER_PATH+'/parsed/'+clouds_data[0][mixed_node[0][i]].split('.')[0] + '.h5'
                    fname_G=clouds_data[0][mixed_node[0][i]].split('.')[1]
                    hf = h5py.File(fname,'r')
                    P = hf[fname_G]
                    if np.shape(P)[0]>args.fine_seg_point_num:
                        tempt1 = np.array(P[:,:13]).T
                        tempt = torch.from_numpy(tempt1.reshape((1,np.shape(tempt1)[0],np.shape(tempt1)[1])))
                        label_modebran = torch.from_numpy(P[:,-1]).long()
                        if args.cuda:
                            tempt = tempt.cuda()
                            label_modebran = label_modebran.cuda()
                        outputsbran = model.FineModule(Variable(tempt.float(), requires_grad=model.training, volatile=not model.training))
                        loss2 = nn.functional.cross_entropy(outputsbran, Variable(label_modebran))
                        _outputsbran,_label_modebran=filter_valid(outputsbran.data.cpu().numpy(),label_modebran.cpu().numpy())
                        loss2.backward()
                        loss2_meter.add(loss2.data[0])
                        acc_meter2.add(_outputsbran,_label_modebran)
                        confusion_matrix.count_predicted_batch_branch(_outputsbran, _label_modebran)

            loss.backward()
            ptnCloudEmbedder.bw_hook()
    
            if args.grad_clip>0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-args.grad_clip, args.grad_clip)
            optimizer.step()
    
            t_trainer = 1000*(time.time()-t0)
            loss_meter.add(loss.data[0]) # pytorch 0.3
    
            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())
            
            if epoch>args.fine_seg_epo & len(mixed_node[0])>0:
                acc_meter.add(o_cpu[Nomixed_node[0],:], t_cpu[[Nomixed_node[0]]])
            else:
                acc_meter.add(o_cpu, t_cpu)
            confusion_matrix.count_predicted_batch(tvec_cpu[[Nomixed_node[0]]], np.argmax(o_cpu[Nomixed_node[0],:],1))
    
            logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.data[0], t_loader, t_trainer)
            t0 = time.time()
    
        return acc_meter.value()[0], meter_value(acc_meter2),loss_meter.value()[0], loss2_meter.value()[0],confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union()
    
    ############
    def eval(epoch):
        start = timer()
        """ Evaluated model on test set """
        model.eval()
    
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
    
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        acc_meter2 = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(5)
    
        # iterate over dataset in batches
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            model.ecc.set_info(GIs, args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
    
            embeddings = ptnCloudEmbedder.run(model, *clouds_data[0:4])
            outputs = model.ecc(embeddings)
            mixed_node=np.where(np.argmax(np.array(outputs.data),axis=1)==5)
            Nomixed_node=np.where(np.argmax(np.array(outputs.data),axis=1)!=5)
            if epoch>args.fine_seg_epo and len(mixed_node[0])>0:
                for i in range(len(mixed_node[0])):
                    fname='/home/data2/qc/large_scalepcss/learning/datasets/tanker/parsed/'+clouds_data[0][mixed_node[0][i]].split('.')[0] + '.h5'
                    fname_G=clouds_data[0][mixed_node[0][i]].split('.')[1]
                    hf = h5py.File(fname,'r')
                    P = hf[fname_G]
                    if np.shape(P)[0]>args.fine_seg_point_num:
                        tempt1 = np.array(P[:,:13]).T
                        tempt = torch.from_numpy(tempt1.reshape((1,np.shape(tempt1)[0],np.shape(tempt1)[1])))
                        label_modebran = torch.from_numpy(P[:,-1]).long()
                        if args.cuda:
                            tempt = tempt.cuda()
                            label_modebran = label_modebran.cuda()
                        outputsbran = model.FineModule(Variable(tempt.float(), requires_grad=model.training, volatile=not model.training))
                        _outputsbran,_label_modebran=filter_valid(outputsbran.data.cpu().numpy(),label_modebran.cpu().numpy())
                        acc_meter2.add(_outputsbran,_label_modebran)
                        confusion_matrix.count_predicted_batch_branch(_outputsbran, _label_modebran)
                        
            o_cpu, t_cpu, tvec_cpu = filter_valid(outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy())
            
            if t_cpu.size > 0:
                if epoch>10 & len(mixed_node[0])>0:
                    acc_meter.add(o_cpu[Nomixed_node[0],:], t_cpu[[Nomixed_node[0]]])
                else:
                    acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu[[Nomixed_node[0]]], np.argmax(o_cpu[Nomixed_node[0],:],1))
        end = timer()
    
        return meter_value(acc_meter), meter_value(acc_meter2),confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union(), confusion_matrix.get_mean_class_accuracy(),confusion_matrix.get_confusion_matrix(), str(end - start)
    
    ############
    def eval_final():
        start = timer()
        """ Evaluated model on test set in an extended way: computes estimates over multiple samples of point clouds and stores predictions """
        model.eval()
        acc_meter2 = tnt.meter.ClassErrorMeter(accuracy=True)
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        confusion_matrix = metrics.ConfusionMatrix(5)
        collected, predictions = defaultdict(list), {}
    
        # collect predictions over multiple sampling seeds
        for ss in range(args.test_multisamp_n):
            test_dataset_ss = create_dataset(args, ss)[1]
            loader = torch.utils.data.DataLoader(test_dataset_ss, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=args.nworkers)
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
    
            # iterate over dataset in batches
            for bidx, (targets, GIs, clouds_data) in enumerate(loader):
                
                model.ecc.set_info(GIs, args.cuda)
                label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
                fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
                embeddings = ptnCloudEmbedder.run(model, *clouds_data[0:4])
                outputs = model.ecc(embeddings)
                
                C=np.where(np.argmax(np.array(outputs.data),axis=1)==5)
                Cother=np.where(np.argmax(np.array(outputs.data),axis=1)!=5)       
                
                for i in range(len(C[0])):
                    fname_G=clouds_data[0][C[0][i]].split('.')[1]
                    hf = h5py.File('/home/data2/qc/large_scalepcss/learning/datasets/tanker/parsed/'+fname + '.h5','r')
                    P = hf[fname_G]
                    if np.shape(P)[0]>2000:
                        tempt1 = np.array(P[:,:13]).T
                        tempt = torch.from_numpy(tempt1.reshape((1,np.shape(tempt1)[0],np.shape(tempt1)[1])))
                        label_modebran = torch.from_numpy(P[:,-1]).long()
                        if args.cuda:
                            tempt = tempt.cuda()
                            label_modebran = label_modebran.cuda()
                        outputsbran = model.FineModule(Variable(tempt.float(), requires_grad=model.training, volatile=not model.training))
                        _outputsbran,_label_modebran=filter_valid(outputsbran.data.cpu().numpy(),label_modebran.cpu().numpy())
                        acc_meter2.add(_outputsbran,_label_modebran)
                        confusion_matrix.count_predicted_batch_branch(_outputsbran, _label_modebran)
  
                if len(C[0])>0:
                    collected[fname].append((outputs.data.cpu().numpy()[Cother[0],:], label_mode_cpu.numpy()[[Cother[0]]], label_vec_cpu.numpy()[Cother[0],:]))
                else:
                    collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))

        # aggregate predictions (mean)
        for fname, lst in collected.items():
            o_cpu, t_cpu, tvec_cpu = list(zip(*lst))
            if args.test_multisamp_n > 1:
                o_cpu = np.mean(np.stack(o_cpu,0),0)
            else:
                o_cpu = o_cpu[0]
            t_cpu, tvec_cpu = t_cpu[0], tvec_cpu[0]
            predictions[fname] = np.argmax(o_cpu,1)
            o_cpu, t_cpu, tvec_cpu = filter_valid(o_cpu, t_cpu, tvec_cpu)
            if t_cpu.size > 0:
                acc_meter.add(o_cpu, t_cpu)
                confusion_matrix.count_predicted_batch(tvec_cpu, np.argmax(o_cpu,1))
    
        per_class_iou = {}
        #perclsiou = confusion_matrix.get_intersection_union_per_class() xuyaohuifu
        #for c, name in dbinfo['inv_class_map'].items():
         #   per_class_iou[name] = perclsiou[c]
        end = timer()
        return meter_value(acc_meter),meter_value(acc_meter2), confusion_matrix.get_overall_accuracy(), confusion_matrix.get_average_intersection_union(), per_class_iou, predictions,  confusion_matrix.get_mean_class_accuracy(), confusion_matrix.get_confusion_matrix(), str(end - start)
    
    ############
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.odir))
        scheduler.step()

        acc, acc_branch, loss,loss2, oacc, avg_iou = train(epoch)

        if (epoch+1) % args.test_nth_epoch == 0 or epoch+1==args.epochs:
            acc_test, acc_test_branch ,oacc_test, avg_iou_test, avg_acc_test, cmatrix,ctime = eval(epoch)
            print('-> Train accuracy: {}, \tTrain_branch accuracy: {}, \tLoss: {}, \tLoss2: {}, \tTest accuracy: {},\tTest_branch accuracy: {},\tTest mAcc: {}, \tTest oAcc: {}, \tTest mIou: {}'.format(acc, acc_branch, loss,loss2,acc_test,acc_test_branch,avg_acc_test, oacc_test, avg_iou_test))
            print('Confusio matrix:')
            print(cmatrix)
            print('Cost Time:')
            print(ctime)
        else:
            acc_test, oacc_test, avg_iou_test, avg_acc_test = 0, 0, 0, 0
            print('-> Train accuracy: {}, \tLoss: {}'.format(acc, loss))

        stats.append({'epoch': epoch, 'acc': acc,'acc_branch': acc_branch, 'loss': loss, 'oacc': oacc, 'avg_iou': avg_iou, 'acc_test': acc_test, 'oacc_test': oacc_test, 'avg_iou_test': avg_iou_test, 'avg_acc_test': avg_acc_test})

        if epoch % args.save_nth_epoch == 0 or epoch==args.epochs-1:
            with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                json.dump(stats, outfile)
            torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                       os.path.join(args.odir, 'model.pth.tar'))

        if math.isnan(loss): break

    if len(stats)>0:
        with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
            json.dump(stats, outfile)

    # Final evaluation
    if args.test_multisamp_n>0:
        acc_test, accbranch_test, oacc_test, avg_iou_test, per_class_iou_test, predictions_test, avg_acc_test, cmatrix2, ctime2 = eval_final()
        print('-> Multisample {}: Test accuracy: {}, \tTest_branch accuracy: {}, \tTest oAcc: {}, \tTest avgIoU: {}, \tTest mAcc: {}'.format(args.test_multisamp_n, acc_test,accbranch_test, oacc_test, avg_iou_test, avg_acc_test))
        print('Confusio matrix:')
        print(cmatrix2)
        print('Cost Time:')
        print(ctime2)
        with h5py.File(os.path.join(args.odir, 'predictions_val.h5'), 'w') as hf:
            for fname, o_cpu in predictions_test.items():
                hf.create_dataset(name=fname, data=o_cpu) #(0-based classes)
        with open(os.path.join(args.odir, 'scores_val.txt'), 'w') as outfile:
            json.dump([{'epoch': args.start_epoch, 'acc_test': acc_test, 'oacc_test': oacc_test, 'avg_iou_test': avg_iou_test, 'per_class_iou_test': per_class_iou_test, 'avg_acc_test': avg_acc_test}], outfile)
        np.save(os.path.join(args.odir, 'pointwise_cm.npy'), cmatrix2)

def resume(args, dbinfo):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    
    checkpoint['args'].model_config = args.model_config #to ensure compatibility with previous arguments convention
    #this should be removed once new models are uploaded
    
    model = create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change
    optimizer = create_optimizer(args, model)
    
    model.load_state_dict(checkpoint['state_dict'])

    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups: group['initial_lr'] = args.lr
    args.start_epoch = checkpoint['epoch']
    try:
        stats = json.loads(open(os.path.join(os.path.dirname(args.resume), 'trainlog.txt')).read())
    except:
        stats = []
    return model, optimizer, stats
    
def create_model(args, dbinfo):
    """ Creates model """
    model = nn.Module()

    nfeat = args.ptn_widths[1][-1]
    model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)
    model.CoarseModule_pn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)
    model.FineModule = fine_grained_module.PointNetSeg(args.ptn_widths_seg[0], args.ptn_widths_seg[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)    
    if args.cuda: 
        model.cuda()
    return model 

def create_optimizer(args, model):
    if args.optim=='sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim=='adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed(seed)    

def filter_valid(output, target, other=None):
    """ Removes predictions for nodes without ground truth """
    idx = target!=-100
    if other is not None:
        return output[idx,:], target[idx], other[idx,...]
    return output[idx,:], target[idx]
    
def meter_value(meter):   
    return meter.value()[0] if meter.n>0 else 0


if __name__ == "__main__": 
    main()
    #eval_final()
