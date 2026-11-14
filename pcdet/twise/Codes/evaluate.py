import argparse
import os
import time
import numpy as np
import imageio
# import matplotlib.pyplot as plt
import scipy.io as io
import torch
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import cv2
from pcdet.twise.Codes.dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from pcdet.twise.Codes.model import *
from pcdet.twise.Codes.metrics import AverageMeter, Result, accum_errorstat
import pcdet.twise.Codes.helper

from pcdet.twise.Codes.utils import *


def iterate(mode, args, loader1, model, optimizer, logger, epoch, curr_step = None):
    
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    SAVE_MODEL_EVERY = 600
    error_binningflag = True
    if curr_step is None:
            curr_step = 0
    
    # switch to appropriate mode
    assert mode in ["train", "val", "test", "val_selection", "test_completion"], \
        "unsupported mode: {}".format(mode)
    
    model.eval()
    lr = 0
    t_total = 0        
        
    ndata_perepoch = round(len(loader1)/args.batch_size)        

    for i, batch_data in enumerate(loader1):

        curr_stepepoch = ndata_perepoch*epoch + curr_step
        batch_data = {
                key: val.to(device)
                for key, val in batch_data.items() if val is not None
            }

        start = time.time()
        data_time = time.time() - start                        

        if mode in  ['test_completion']:                

            SAVE_SAMPLEINTERV = 1

            color_img = batch_data['rgb']


            sparse_dep = batch_data['d']/params['depth_maxrange']
            sparse_C = (sparse_dep > 0).float()

            valpred_dc, __, __ = model(sparse_dep, color_img)
            pred_datavalid = smooth2chandep(valpred_dc, params = params, device = device)
            #pred_datavalid = hard2chandep(valpred_dc, params = params, device = device)
            #pred_datavalid =  F.relu(valpred_dc)*params['depth_maxrange']

            if not curr_step%SAVE_SAMPLEINTERV:
                pred_dep = pred_datavalid.detach().cpu().numpy()
                pred_dep = np.squeeze(pred_dep)
                pred_h, pred_w = np.shape(pred_dep)   
                zero_mtrx = np.zeros((352 - pred_h, owidth))
                pred_dep = np.concatenate((zero_mtrx, pred_dep))
                pred_dep_round = np.uint16(pred_dep*256)
                path_i = str(curr_step).zfill(6) + '.png'

                print('Finished writing sample %06d.png\n'%curr_step)
            curr_step += 1                    
            avg = None
            is_best = None
    
    return avg, is_best

def eval():

    checkpoint = None
    is_eval = False


    model = MultiRes_network_avgpool_diffspatialsizes().to(device)


    model = torch.nn.DataParallel(model)


    print("=> creating data loaders ... ")
    val_dataset = KittiDepth("test_completion", args, trainvalsplit = 0.95)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder

    if checkpoint is not None:
        print("=> checkpoint is None.")
    print("=> logger created.")    

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate('val_selection' if not args.test else 'test_completion', args, val_loader, model, None, logger, 
                                  checkpoint['epoch'])
