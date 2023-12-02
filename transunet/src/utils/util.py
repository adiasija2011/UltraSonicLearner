import argparse
import os

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_image(x, norm=False):
    if norm:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        x = x * 255
        x = x.detach().cpu().numpy()
        for i in range(x.shape[0]):
            x[i, :, :] = (x[i, :, :] * std[i]) + mean[i]
    else:
        x = x.detach().cpu().numpy()
    x = x * 255
    x = x.astype(np.uint8)
    x = x.transpose(1, 2, 0)
    return x
def save_results(inps, preds, tgts, size=256, dir='results/', batch_idx=0):
    batch_idx = str(batch_idx)
    preds = torch.sigmoid(preds)
    preds = preds > 0.5
    if not os.path.exists(dir):
        os.mkdir(dir)
    img_idx = 0
    for (inp, pred, tgt) in zip(inps, preds, tgts):
        inp = to_image(inp, norm=True)

        # inp = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR)
        pred = to_image(pred)
        # pred = np.concatenate([pred, pred, pred], axis=2)
        tgt = to_image(tgt)
        # tgt = np.concatenate([tgt, tgt, tgt], axis=2)
        frame = np.zeros((size, size * 3, 1))
        frame[:, :size, :] = inp[:, :, 0][:, :, np.newaxis]
        frame[:, size:size * 2, :] = pred
        frame[:, size * 2:size * 3, :] = tgt
        name = dir + batch_idx + '_' + str(img_idx) + '.png'
        cv2.imwrite(name, frame)
        img_idx += 1
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
