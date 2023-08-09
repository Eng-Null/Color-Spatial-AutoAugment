import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision

from tqdm import tqdm

import Data
import Loss
import itertools
from models.resnet import ResNet18, ResNet50
from models.wide_resnet  import WideResNet2810
from Optimizer import *
from utils import AverageMeter, save_model, initialize_dir
from config import arg_parser

def load_model(args):
    if(args.network =='resnet18'):
        model = ResNet18()
    elif(args.network == 'resnet50'):
        model = ResNet50()
    elif(args.network == 'wideresnet'):
        model = WideResNet2810()

    if args.device == 'cuda':
        model = model.cuda()
        #model = nn.DataParallel(model)
        cudnn.benchmark = True

    return model

def load_loss(args):
    return Loss.InfoNCELoss(tau=args.tau)

def train(train_loader, model, criterion, optimizer, epoch, args):
    train_loss = AverageMeter()

    model.train()

    train_bar = tqdm(train_loader, ncols=100)
    for imagei, imagej, _ in train_bar:

        imagei = imagei.to(args.device)
        imagej = imagej.to(args.device)
        
        _, outputi = model(imagei)
        _, outputj = model(imagej)

        loss = criterion(outputi, outputj)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), imagei.size(0))
        
        # Print informations
        train_bar.set_description(f'Train: [{epoch+1}|{args.epochs}] '+
                    f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                    f'Lr: {optimizer.param_groups[0]["lr"]:.6f} ')

    return train_loss.avg

def validate(val_loader, model, criterion, epoch, args):
    val_loss = AverageMeter()

    model.eval()

    with torch.no_grad():
        val_bar = tqdm(val_loader, ncols=100)
        for imagei, imagej, _ in val_bar:
            
            imagei = imagei.to(args.device)
            imagej = imagej.to(args.device)

            _, outputi = model(imagei)
            _, outputj = model(imagej)

            loss = criterion(outputi, outputj)

            val_loss.update(loss.item(), imagei.size(0))

            # Print informations
            val_bar.set_description(f'Valid: [{epoch+1}|{args.epochs}] '
                        f'Loss: {val_loss.val:.4f} ({val_loss.avg:.4f})')

    return val_loss.avg

if __name__ == '__main__':
    args = arg_parser()
    
    best_loss = float(99.99)

    # data loader
    print(Data.get_mean_std(args))
    transform = Data.Transform_train_Cifar10(args = args)
    train_loader, val_loader = transform.load_data()

    batch = list(itertools.islice(train_loader, 1))
    example_datai, _, _ = batch[0]
    img_gridi = torchvision.utils.make_grid(example_datai)

    # model + loss
    model = load_model(args)
    criterion = load_loss(args)

    # optimizer + lr scheduler
    optimizer = load_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    
    # initialize checkpoint directory
    initialize_dir('./checkpoint/{}_{}_{}'.format(args.log_dir, args.dataset, args.augment))
    # tensorboardX
    initialize_dir('./tensorboard/{}_{}_{}'.format(args.log_dir, args.dataset, args.augment))
    #os.makedirs('./tensorboard/{}_{}_{}'.format(args.log_dir, args.dataset, args.augment), exist_ok=True)
    summary_writer = SummaryWriter('./tensorboard/{}_{}_{}'.format(args.log_dir, args.dataset, args.augment))

    summary_writer.add_image('{}_batch_example'.format(args.dataset), img_gridi)
    example_datai = example_datai.to(args.device)
    summary_writer.add_graph(model, example_datai)
    # summary model
    summary(model, input_size=(3, 32, 32))

    for epoch in range(args.epochs):
        # train
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # validate
        #val_loss = validate(val_loader, model, criterion, epoch, args)

        # tensorboardX
        summary_writer.add_scalars('loss', {'Train': train_loss,
                                       #'Validation': val_loss
                                        }, 
                                       epoch+1)
        
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch+1)
        summary_writer.add_text('lr', str(optimizer.param_groups[0]['lr']), epoch+1)
        summary_writer.add_text('train_loss', str(train_loss), epoch+1)
        #summary_writer.add_text('val_loss', str(val_loss), epoch+1)

        #best_loss = min(best_loss, val_loss)

        if (epoch+1) % args.checkpoint_freq == 0:
            save_model(model, optimizer, epoch+1, lr_scheduler, './checkpoint/{}_{}_{}'.format(args.log_dir, args.dataset, args.augment) + f'/ckpt_epoch_{epoch+1}.pth', args)

        lr_scheduler.step()

    save_model(model, optimizer, epoch+1, lr_scheduler, 'trained_final_{}_{}_{}.pth'.format(args.epochs, args.augment, args.dataset), args)

    summary_writer.close()
