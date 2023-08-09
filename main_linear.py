import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from tqdm import tqdm

from models.resnet import ResNet18, ResNet18Linear
from models.resnet import ResNet50, ResNet50Linear
from models.wide_resnet  import WideResNet2810, WideResNet2810Linear
import Data
import itertools
from Optimizer import load_optimizer, get_lr_scheduler
from utils import AverageMeter, accuracy, save_model, initialize_dir
from config import arg_parser

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_model(args):
    if(args.network =='resnet18'):
        model = ResNet18()
        model_linear = ResNet18Linear()
    elif(args.network == 'resnet50'):
        model = ResNet50()
        model_linear = ResNet50Linear()
    elif(args.network == 'wideresnet'):
        model = WideResNet2810()
        model_linear = WideResNet2810Linear()

    if args.device == 'cuda':
        model = model.cuda()
        model_linear = model_linear.cuda()
        cudnn.benchmark = True

    pth = torch.load('trained_final_1000_AutoAugment_cifar10.pth')
    state_dict = pth['model']
    model.load_state_dict(state_dict)

    return model, model_linear

def load_loss(args):
    return nn.CrossEntropyLoss()

def train(train_loader, model, model_linear, criterion, optimizer, epoch, args):
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # embedding network remains unchanged while training linear model
    model.eval()
    model_linear.train()

    train_bar = tqdm(train_loader, ncols=150)
    for images, labels in train_bar:

        images = images.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            feature = model.encoder(images)
        output = model_linear(feature)
        loss = criterion(output, labels)

        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), images.size(0))

        # Print informations
        train_bar.set_description(f'Train: [{epoch+1}|{args.epochs_linear}] '+
                    f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                    f'prec5: {top5.val:.3f} ({top5.avg:.3f}) '
                    f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} ')

    return train_loss.avg, top1.avg, top5.avg

def validate(val_loader, model, model_linear, criterion, epoch, args):
    val_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    model_linear.eval()

    with torch.no_grad():
        val_bar = tqdm(val_loader, ncols=150)
        y_pred = [] # save predction
        y_true = [] # save ground truth
        # constant for classes
        classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
        for images, labels in val_bar:
            images = images.to(args.device)
            labels = labels.to(args.device)

            output = model_linear(model.encoder(images))
            loss = criterion(output, labels)

            prec1, prec5 = accuracy(output, labels, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            val_loss.update(loss.item(), images.size(0))
            
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # save prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth

            # Print informations
            val_bar.set_description(f'Valid: [{epoch+1}|{args.epochs_linear}] '
                        f'prec1: {top1.val:.3f} ({top1.avg:.3f}) '
                        f'prec5: {top5.val:.3f} ({top5.avg:.3f}) '
                        f'Loss: {val_loss.val:.4f} ({val_loss.avg:.4f})')
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(16, 9))    
    # Save confusion matrix to Tensorboard
    summary_writer.add_figure("Confusion matrix", sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}).get_figure(), epoch)

    return val_loss.avg, top1.avg, top5.avg

if __name__ == '__main__':
    args = arg_parser()
    
    best_acc = 0

    # data loader
    transform = Data.Transform_train_Cifar10(args = args)
    train_loader, val_loader = transform.load_Single_data()

    batch = list(itertools.islice(train_loader, 1))
    example_data, _ = batch[0]
    img_gridi = torchvision.utils.make_grid(example_data)

    # model + loss
    model, model_linear = load_model(args)
    criterion = load_loss(args)

    # optimizer + lr scheduler
    optimizer = load_optimizer(args, model_linear)
    lr_scheduler = get_lr_scheduler(args, optimizer)
  
    # initialize checkpoint directory
    initialize_dir('./checkpoint/{}_{}_{}_linear'.format(args.log_dir, args.dataset, args.augment))
    
    # tensorboardX
    initialize_dir('./tensorboard/{}_{}_{}_linear'.format(args.log_dir, args.dataset, args.augment))

    # tensorboardX
    summary_writer = SummaryWriter('./tensorboard/{}_{}_{}_linear'.format(args.log_dir,args.dataset, args.augment))
    
    summary_writer.add_image('{}_batch_example'.format(args.dataset), img_gridi)

    for epoch in range(args.epochs_linear):
        # train
        train_loss, train_acc_top1, train_acc_top5 = train(train_loader, model, model_linear, criterion, optimizer, epoch, args)

        # validate
        val_loss, val_acc_top1, val_acc_top5, = validate(val_loader, model, model_linear, criterion, epoch, args)

        # tensorboardX
        summary_writer.add_scalars('loss__linear', {'Train': train_loss,
                                               'Validation': val_loss}, epoch+1)
        
        summary_writer.add_scalars('accuracy__linear', {'Train_Top1': train_acc_top1,
                                                        'Train_Top5': train_acc_top5,
                                                   'Validation_Top1': val_acc_top1,
                                                   'Validation_Top5': val_acc_top5}, epoch+1)
        
        summary_writer.add_scalar('lr_linear', optimizer.param_groups[0]['lr'], epoch+1)
        summary_writer.add_text('lr_linear', str(optimizer.param_groups[0]['lr']), epoch+1)
        summary_writer.add_text('train_loss_linear', str(train_loss), epoch+1)
        summary_writer.add_text('train_acc_top1', str(train_acc_top1), epoch+1)
        summary_writer.add_text('train_acc_top5', str(train_acc_top5), epoch+1)
        summary_writer.add_text('val_loss_linear', str(val_loss), epoch+1)
        summary_writer.add_text('val_acc_top1', str(val_acc_top1), epoch+1)
        summary_writer.add_text('val_acc_top5', str(val_acc_top5), epoch+1)

        best_acc = max(best_acc, val_acc_top1)


        if epoch % args.checkpoint_freq_linear == 0:
            save_model(model_linear, optimizer, epoch+1, lr_scheduler, './checkpoint/{}_{}_{}_linear'.format(args.log_dir, args.dataset, args.augment) + f'/ckpt_linear_epoch_{epoch+1}.pth', args)

        lr_scheduler.step()

    print('Best accuracy:', best_acc)

    save_model(model_linear, optimizer, epoch+1, lr_scheduler, 'trained_final_Linear.pth', args)

    summary_writer.close()
