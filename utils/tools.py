import os
import time
import random

import math
import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms


# 固定随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 定义统计汇总的类型
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


# 跟踪某个标量指标（如 loss、acc）的 当前值/累计和/次数/平均值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    # 打印第 i 个 batch 的时间、loss、acc 等
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    # 在整个验证循环结束后打印汇总
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    # 生成进度条样式
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


import ipdb


# 计算 top-k 准确率
# top1: 相似度最高的一个与真实标签一样就算对
# top5: 相似度最高的前5个与真实标签有一个一样就算对
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 从 checkpoint 加载 state_dict 并恢复训练状态
def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device is not 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # model.prompt_generator-->DMN 可能将“可学习的提示/记忆模块”独立成子模块，便于只加载/微调这部分参数
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


# 在验证集上评估模型，返回 top1 平均精度
def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if hasattr(model, 'branch4_logits') and getattr(args, 'use_bridge_branch', False):
                    logits_b4 = model.branch4_logits(images)  # [B, C]
                    # 与主输出相加（若主输出已是三分支融合结果）
                    output = output + args.bridge_w * logits_b4
                # === 结束 ===

                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg
