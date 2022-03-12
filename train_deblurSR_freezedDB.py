import os
from config import Config
from torch.utils.tensorboard import SummaryWriter

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data_joint
from MPRNet_SR import MPRNet
import losses
from tqdm import tqdm
from pdb import set_trace as stx


start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR

######### Model ###########
model_restoration = MPRNet()
model_restoration.cuda()


path_chk_rest1 = utils.get_last_path(model_dir, '_latest.pth')
utils.load_checkpoint(model_restoration, path_chk_rest1)

for k, v in model_restoration.named_parameters():
    if v.requires_grad:
        if 'SRNet_' not in k:
            print("Grad=False: ", k)
            v.requires_grad = False
        else:
            print("Grad=True: ", k)


device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


# Create parameter groups for joint training
start_lr_deblur = opt.OPTIM.LR_INITIAL_DB
start_lr = opt.OPTIM.LR_INITIAL
if start_lr_deblur > 0:
    sr_params = []
    predeblur_params = []
    for k, v in model_restoration.named_parameters():
        if v.requires_grad:
            if 'SRNet_' not in k:
                predeblur_params.append(v)
                print("deblur: ", k)
            else:
                sr_params.append(v)
                print("SR: ", k)
        else:
            print('Warning: Params [{:s}] will not optimize.'.format(k))

    optim_params = [
        {
            'params': sr_params,
            'lr': start_lr
        }
    ]
    # print("optimizing only this params: ", optim_params)
    optimizer = optim.Adam(optim_params, lr=start_lr, betas=(0.9, 0.999), eps=1e-8)
else:
    optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)
print("LR_INITIAL = ", start_lr)
# print("LR_INITIAL_DB = ", start_lr_deblur)
print("LR_MIN = ", opt.OPTIM.LR_MIN)


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for k, v in model_restoration.named_parameters():
        if v.requires_grad:
            if 'SRNet_' not in k:
                print("Grad=False: ", k)
                v.requires_grad = False
            else:
                print("Grad=True: ", k)

    for k, v in model_restoration.named_parameters():
        if not v.requires_grad:
            print('Warning: Params [{:s}] will not optimize.'.format(k))

    print ("loaded start lr = ", optimizer.param_groups[0]['lr'])
    # print("loaded start lr SR = ", optimizer.param_groups[1]['lr'])
    optimizer.param_groups[0]['lr'] = start_lr
    # optimizer.param_groups[1]['lr'] = start_lr
    print ("after reset start lr = ", optimizer.param_groups[0]['lr'])
    # print("after reset start lr SR = ", optimizer.param_groups[1]['lr'])

    print("start epoch = ", start_epoch)

    for i in range(1, start_epoch): # start_epoch-1???
        scheduler.step()
    # start_epoch = 1
    start_lr = scheduler.get_last_lr()[0]
    # start_lr = scheduler.get_last_lr()[1]
    print("after step scheduler_lr = ", start_lr)
    # print("after step scheduler_lr_SR = ", start_lr)

    print("after step optimizer_lr = ", optimizer.param_groups[0]['lr'])
    # print("after step optimizer_lr(SR) = ", optimizer.param_groups[1]['lr'])

    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with:")
    print("-- learning rate:", start_lr)
    # print("-- learning rate SR:", start_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
# criterion_char = losses.CharbonnierLoss()
# criterion_edge = losses.EdgeLoss()
criterion_l1 = torch.nn.L1Loss().to('cuda')
criterion_grad = losses.Gradient_Loss(device='cuda').to('cuda')

######### DataLoaders ###########
print("Creating train dataloader...")
train_dataset = get_training_data(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)

print("Creating test dataloder...")
val_dataset = get_validation_data_joint(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

writer = SummaryWriter(log_dir='summary', comment=f'LR_{opt.OPTIM.LR_INITIAL}_BS_{opt.OPTIM.BATCH_SIZE}')

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    input_ = None
    restored_dbs = None
    restored_sr = None

    # writer.add_scalar('learning_rate deblur', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('learning_rate SR', scheduler.get_last_lr()[0], epoch)

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # if i == 51:
        #     break
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        input_ = data[0].cuda()
        target_db = data[1].cuda()
        target_hr = data[2].cuda()

        restored_dbs, restored_sr = model_restoration(input_)

        if i % 150 == 0:
            writer.add_images('train/input', input_, (epoch - 1)*len(train_loader) + i)
            writer.add_images('train/target_db', target_db, (epoch - 1)*len(train_loader) + i)
            writer.add_images('train/target_hr', target_hr, (epoch - 1)*len(train_loader) + i)
            writer.add_images('train/pred/hr', restored_sr, (epoch - 1)*len(train_loader) + i)
            writer.add_images('train/pred/lr', restored_dbs[0], (epoch - 1)*len(train_loader) + i)


        # Compute loss at each stage
        # loss_char = criterion_char(restored_dbs[1],target_db) + criterion_char(restored_dbs[2],target_db)
        # print(loss_char)

        # loss_edge = criterion_edge(restored_dbs[1],target_db) + criterion_edge(restored_dbs[2],target_db)
        # print("loss_Edge: ", loss_edge)

        loss_l1 = criterion_l1(restored_sr, target_hr)
        loss_grad = criterion_grad(restored_sr, target_hr)
        loss_sr = loss_l1 + (0.1*loss_grad)
        # print("Loss_sr: ", loss_sr)

        # loss_db = 0.5 * (loss_char + (0.05 * loss_edge))
        # loss = loss_db + loss_sr
        loss = loss_sr
        # print(loss)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 50 == 0:
            writer.add_scalar('Loss/train', loss.item(), (epoch - 1)*len(train_loader) + i)
            # writer.add_scalar('Loss_db/train', loss_db.item(), (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('Loss_sr/train', loss_sr.item(), (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('Loss_l1/train', loss_l1.item(), (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('Loss_grad/train', loss_grad.item(), (epoch - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss_edge/train', loss_edge.item(), (epoch - 1) * len(train_loader) + i)
            # writer.add_scalar('Loss_char/train', loss_char.item(), (epoch - 1) * len(train_loader) + i)

    epoch_loss /= len(train_loader)

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))
    print("Saved model {}".format(epoch, "model_epoch_{}.pth".format(epoch)))

    #### Evaluation ####
    if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        val_loss = 0
        psnr_score = 0
        psnr_score_low = 0

        target = None

        for ii, data_val in enumerate(tqdm(val_loader), 0):
            input_ = data_val[0].cuda()
            target_low = data_val[1].cuda()
            target = data_val[2].cuda()

            with torch.no_grad():
                restored_dbs, restored_sr = model_restoration(input_)

            val_loss += torch.nn.MSELoss()(restored_sr, target)
            psnr_score_low += utils.psnr(restored_dbs[0], target_low)
            psnr_score += utils.psnr(restored_sr, target)

        val_loss /= len(val_loader)
        psnr_score_low /= len(val_loader)
        psnr_score /= len(val_loader)

        if psnr_score > best_psnr:
            best_psnr = psnr_score
            best_epoch = epoch
            # torch.save({'epoch': epoch,
            #             'state_dict': model_restoration.state_dict(),
            #             'optimizer' : optimizer.state_dict()
            #             }, os.path.join(model_dir,"model_best_epoch_{}.pth".format(epoch)))
            # print("Saved best model: epoch {}: model: {}".format(epoch, "model_best_epoch_{}.pth".format(epoch)))

        print("[epoch %d Loss: %.4f PSNR_DB: %.4f PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, val_loss, psnr_score_low, psnr_score, best_epoch, best_psnr))

        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Validation_Score', psnr_score, epoch)
        writer.add_scalar('Validation_Score_DB', psnr_score_low, epoch)
        writer.add_images('test/input', input_, epoch)
        writer.add_images('test/target', target, epoch)
        writer.add_images('test/pred/LR', restored_dbs[0], epoch)
        writer.add_images('test/pred/HR', restored_sr, epoch)

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.6f}\tLearningRate_SR {:.6f}".format(
           epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]))
    print("------------------------------------------------------------------")

writer.close()
