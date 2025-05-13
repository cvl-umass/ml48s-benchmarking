import argparse
import os
import random
import shutil
import warnings
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision

from sklearn.metrics import precision_recall_curve, average_precision_score

from datasets.dataset import get_dataset, wrap_dataset
from losses.get_loss import get_loss_func
from meters import AverageMeter, ProgressMeter, Summary
from losses.ChecklistReg import ChecklistReg

def parse_args(argstring=None):
    model_names = ["resnet50"]
    dataset_names = ['COCO2014', 'L48']

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data-dir', metavar='DIR', nargs='?', default="data/")
    parser.add_argument('--dataset', choices=dataset_names, default="COCO2014")
    parser.add_argument('--output-dir', default="out/", help='path to checkpoints')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--epochs', default=10, type=int, metavar='N')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--optimizer', default='adam', type=str,choices=['sgd', 'adam'],)
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',)
    parser.add_argument('--weight-decay', default=0, type=float,)
    parser.add_argument('-p', '--print-freq', default=100, type=int,)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',)
    parser.add_argument('--old-weights', action=argparse.BooleanOptionalAction, default=True)
    
    # Dataset removals, for training data
    parser.add_argument('--remove-empty', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--remove-unannotated', action=argparse.BooleanOptionalAction, default=True)
    
    # Training variant
    parser.add_argument("--train-set-variant", type=str, default="full")
    parser.add_argument("--val-set-variant", type=str, default="full")
    parser.add_argument('--val-frac', default=0.2, type=float)
    parser.add_argument('--split-seed', default=1200, type=int)
    
    # Unique Loss functions
    parser.add_argument("--mlab-loss-func", type=str, default="BCE")
    parser.add_argument("--neg-prior-a", type=float, default=1)
    parser.add_argument("--neg-prior-b", type=float, default=0)
    # LS
    parser.add_argument("--lsl-eps", type=float, default=0.1)
    # WAN
    parser.add_argument("--wnl-gamma", type=float, default=None)
    # LL
    parser.add_argument("--ll-mode", type=str, default='rejection')
    parser.add_argument("--ll-delta", type=float, default=0.2)
    parser.add_argument("--linear-lr-mult", type=float, default=1)
    # EM
    parser.add_argument("--eml-alpha", type=float, default=0.1)

    # ROLE
    parser.add_argument("--epl-lamb", type=float, default=1)
    parser.add_argument("--epl-k", type=float, default=None)
    parser.add_argument("--role-lr-mult", type=float, default=10)

    # Checklist regularization
    parser.add_argument('--checklist-reg', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--checklist-reg-alpha', type=float, default=0.0)
    parser.add_argument('--checklist-reg-eps', type=float, default=0.001)
    parser.add_argument('--checklist-reg-init-val', type=float, default=None)

    # Logging stuff
    parser.add_argument('--job-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    if argstring is not None:
        import shlex
        args = parser.parse_args(shlex.split(argstring))
    else:
        args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    num_classes = {
        'L48': 100,
        'COCO2014': 80,
    }
    avg_positives = {
        'L48': 1.6 if args.remove_empty else (1.4 if args.remove_unannotated else 0.81),
        'COCO2014': 2.9,
    }
    args.num_classes = num_classes[args.dataset]
    args.epl_k = avg_positives[args.dataset]
    return args

def main():
    args = parse_args()
    print(args)

    global best_val_map, best_test_map
    
    if not os.path.isdir(os.path.join(args.output_dir, str(args.job_id))):
        os.makedirs(os.path.join(args.output_dir, str(args.job_id)))

    if args.pretrained:
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1' if args.old_weights else "DEFAULT")
        model.fc = torch.nn.Linear(2048, args.num_classes)
    else:
        model = torchvision.models.resnet50(weights=None, num_classes=args.num_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # define loss function (criterion), optimizer
    criterion = get_loss_func(args.mlab_loss_func, args, model=model).to(device)
    optimizer = get_optimizer(args, model)

    best_val_map = {'average_precision': {'macro': 0}}
    best_test_map = {'average_precision': {'macro': 0}}

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            else:
                checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_val_map, best_test_map = checkpoint['best_val_map'], checkpoint['best_test_map']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        if args.confirm_epochs and 'best' in args.resume:
            args.start_epoch = args.epochs

    datasets = get_dataset(args)
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']

    train_loader = wrap_dataset(train_dataset, "train", args)
    val_loader = wrap_dataset(val_dataset, "val", args)
    test_loader = wrap_dataset(test_dataset, "test", args)

    if args.checklist_reg:
        reg = ChecklistReg(criterion, train_dataset, alpha=args.checklist_reg_alpha, eps=args.checklist_reg_eps, init_val=args.checklist_reg_init_val)
    else:
        reg = None

    if args.mlab_loss_func == 'ROLE':
        model.g = torch.rand(len(train_dataset), args.num_classes) * (np.log(0.6 / (1 - 0.6)) - np.log(0.4 / (1 - 0.4))) + np.log(0.4 / (1 - 0.4))

        full_labels = train_dataset.label_matrix
        annotated_masks = train_dataset.mask

        known_positives = full_labels * annotated_masks # Only 1 * 1 = 1, so known positives require mask = 1 and label = 1
        model.g[known_positives > 0] = np.log(0.995 / (1 - 0.995))

        model.g = torch.nn.Parameter(model.g.to(device))
        optimizer.add_param_group({'params': model.g, 'lr': args.lr * args.role_lr_mult})

    if args.evaluate:
        val_map = validate(args, 'val', device, model, val_loader, criterion)
        test_map = validate(args, 'test', device, model, test_loader, criterion)
        return

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        train_one_epoch(args, epoch, device, model, train_loader, criterion, reg, optimizer)
        val_map = validate(args, 'val', device, model, val_loader, criterion)
        test_map = validate(args, 'test', device, model, test_loader, criterion)

        is_best = val_map["average_precision"]["full_macro"] > best_val_map["average_precision"]["full_macro"]
        if is_best:
            best_val_map = val_map
            best_test_map = test_map

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_val_map': best_val_map,
            'best_test_map': best_test_map,
        }, is_best, args, f"{args.job_id}_{epoch+1:03d}.pth.tar")


def train_one_epoch(args, epoch, device, model, train_loader, criterion, reg, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = data['image'].to(device, non_blocking=True)
        target = data['labels'].to(device, non_blocking=True)
        mask = data['mask'].to(device, non_blocking=True)

        target[mask == 0] = 0

        # compute output
        output = model(images)
        if args.mlab_loss_func == 'ROLE':
            loss = criterion(output, target, mask, model.g[data['idx']])
        elif args.mlab_loss_func == 'LL' and args.ll_mode == 'perm_correction':
            loss, correction_idx = criterion(output, target, mask)
            train_loader.dataset.label_matrix[data['idx'][correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0
            train_loader.dataset.mask[data['idx'][correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0
        else:
            loss = criterion(output, target, mask)

        known_negs = (mask == 1) & (target == 0)
        loss[known_negs] = loss[known_negs] * args.neg_prior_a
        loss[known_negs] = loss[known_negs] + args.neg_prior_b * nn.functional.binary_cross_entropy_with_logits(output[known_negs], target[known_negs], reduce='none')

        loss = loss.mean()
        if reg is not None:
            loss = loss + reg(output, data['idx'])

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    progress.display_summary()
    return losses.avg

def validate(args, split, device, model, val_loader, criterion):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses,],
        prefix=f'{split}: ')

    # switch to evaluate mode
    model.eval()

    def run_validate(loader):
        with torch.no_grad():
            end = time.time()
            gt_presence = []
            predicted = []
            paths = []
            masks = []
            for i, data in enumerate(loader):
                images = data['image'].to(device, non_blocking=True)
                target = data['labels'].to(device, non_blocking=True)
                mask = data['mask'].to(device, non_blocking=True)

                target[mask == 0] = 0
                
                output = model(images)
                loss = criterion(output, target, mask)
                losses.update(loss.item(), images.size(0))

                for t in target:
                    gt_presence.append(t.cpu())
                for out in output:
                    predicted.append(torch.sigmoid(out).cpu())
                paths.extend(data['path'])
                masks.extend(mask.cpu())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)
            return gt_presence, predicted, paths, masks

    gt_presence, predicted, paths, masks = run_validate(val_loader)

    precision = run_stats(gt_presence, predicted, masks)

    progress.display_summary()
    print(f'{split} Macro AP (full): {precision["average_precision"]["full_macro"]}')
    return precision

def run_stats(gt_presence, predicted, masks):
    """
    Get precision/recall stats where gt_presence is 1/0 and predicted is the scores. 
    Both shapes are (num_datapoints, num_classes)
    """
    assert gt_presence.shape == predicted.shape
    assert gt_presence.shape == masks.shape
    num_classes = gt_presence.shape[1]
    gt_presence = gt_presence.numpy()
    predicted = predicted.numpy()
    masks = masks.numpy()

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        annotated = np.all(masks > 0, axis=1)
        if annotated.sum() > 0:
            precision[f'full_{i}'], recall[f'full_{i}'], _ = precision_recall_curve(gt_presence[annotated, i], predicted[annotated, i])
            average_precision[f'full_{i}'] = average_precision_score(gt_presence[annotated, i], predicted[annotated, i])

    average_precision["full_macro"] = np.mean([average_precision[f'full_{i}'] for i in range(num_classes) if f'full_{i}' in average_precision])
    return {"precision": precision, "recall": recall, "average_precision": average_precision}

def get_optimizer(args, model):
    fc_params_names = ['module.fc.weight', 'module.fc.bias']
    fc_params = list(filter(lambda kv: kv[0] in fc_params_names, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in fc_params_names, model.named_parameters()))
    opt_params = [
        {'params': [temp[1] for temp in base_params], 'lr' : args.lr},
        {'params': [temp[1] for temp in fc_params], 'lr' : args.lr * args.linear_lr_mult}
    ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(opt_params,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(opt_params,
                                weight_decay=args.weight_decay)
    return optimizer

def save_checkpoint(state, is_best, args, filename):
    out_dir = os.path.join(args.output_dir, str(args.job_id))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        for fname in ['predicted', 'paths', 'ground_truth', 'masks', 'precision', 'asset_embedding', 'running_avg_pred', 'recall_at_p95']:
            if os.path.exists(os.path.join(out_dir, f'{fname}.pth')):
                shutil.copyfile(os.path.join(out_dir, f'{fname}.pth'), os.path.join(out_dir, f'{fname}_best.pth'))
        shutil.copyfile(os.path.join(out_dir, filename), os.path.join(out_dir, 'model_best.pth.tar'))
    os.remove(os.path.join(out_dir, filename))


if __name__ == '__main__':
    main()
