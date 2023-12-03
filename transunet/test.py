import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score
from src.utils.util import save_results

from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="U_Net",
                    choices=["TransUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--test_file_dir', type=str, default="busi_test.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=300, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=41, help='random seed')
parser.add_argument('--ckpt', type=str, default="./checkpoint/TransUnet_model.pth", help='checkpoint')
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    model = get_transformer_based_model(model_name=args.model, img_size=args.img_size,
                                        num_classes=args.num_classes, in_ch=3).cuda()
    return model


def getDataloader(args):
    img_size = args.img_size
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_val = MedicalDataSets(base_dir=args.base_dir, split="test", transform=val_transform,
                             test_file_dir=args.test_file_dir)
    print("test num:{}".format(len(db_val)))

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4)

    return valloader


def main(args):
    valloader = getDataloader(args=args)

    model = get_model(args)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))

    print("test file dir:{}".format(args.test_file_dir))

    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    print("{} test samples".format(len(valloader)))

    avg_meters = {'val_loss': AverageMeter(),
                  'val_iou': AverageMeter(),
                  'val_dice': AverageMeter(),
                  'val_SE': AverageMeter(),
                  'val_PC': AverageMeter(),
                  'val_F1': AverageMeter(),
                  'val_ACC': AverageMeter()}

    avg_meters_benign = {'val_loss': AverageMeter(),
                         'val_iou': AverageMeter(),
                         'val_dice': AverageMeter(),
                         'val_SE': AverageMeter(),
                         'val_PC': AverageMeter(),
                         'val_F1': AverageMeter(),
                         'val_ACC': AverageMeter()}

    avg_meters_malignant = {'val_loss': AverageMeter(),
                            'val_iou': AverageMeter(),
                            'val_dice': AverageMeter(),
                            'val_SE': AverageMeter(),
                            'val_PC': AverageMeter(),
                            'val_F1': AverageMeter(),
                            'val_ACC': AverageMeter()}

    model.eval()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            case = sampled_batch['case'][0]
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
            output = model(img_batch)
            loss = criterion(output, label_batch)
            save_results(img_batch, output, label_batch, batch_idx=i_batch)
            iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)

            avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
            avg_meters['val_iou'].update(iou, img_batch.size(0))
            avg_meters['val_dice'].update(dice, img_batch.size(0))
            avg_meters['val_SE'].update(SE, img_batch.size(0))
            avg_meters['val_PC'].update(PC, img_batch.size(0))
            avg_meters['val_F1'].update(F1, img_batch.size(0))
            avg_meters['val_ACC'].update(ACC, img_batch.size(0))

            if 'benign' in case:
                avg_meters_benign['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters_benign['val_iou'].update(iou, img_batch.size(0))
                avg_meters_benign['val_dice'].update(dice, img_batch.size(0))
                avg_meters_benign['val_SE'].update(SE, img_batch.size(0))
                avg_meters_benign['val_PC'].update(PC, img_batch.size(0))
                avg_meters_benign['val_F1'].update(F1, img_batch.size(0))
                avg_meters_benign['val_ACC'].update(ACC, img_batch.size(0))

            elif 'malignant' in case:
                avg_meters_malignant['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters_malignant['val_iou'].update(iou, img_batch.size(0))
                avg_meters_malignant['val_dice'].update(dice, img_batch.size(0))
                avg_meters_malignant['val_SE'].update(SE, img_batch.size(0))
                avg_meters_malignant['val_PC'].update(PC, img_batch.size(0))
                avg_meters_malignant['val_F1'].update(F1, img_batch.size(0))
                avg_meters_malignant['val_ACC'].update(ACC, img_batch.size(0))

    print('Overall Test_loss %.4f - Overall Test_iou %.4f - Overall Test_dice %.4f - Overall Test_SE %.4f - '
          'Test_PC %.4f - Test_F1 %.4f - Test_ACC %.4f '
          % (
          avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg, avg_meters['val_SE'].avg,
          avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg))

    print('Benign Test_loss %.4f - Benign Test_iou %.4f - Benign Test_dice %.4f - Benign Test_SE %.4f - '
          'Test_PC %.4f - Test_F1 %.4f - Test_ACC %.4f '
          % (avg_meters_benign['val_loss'].avg, avg_meters_benign['val_iou'].avg,
             avg_meters_benign['val_dice'].avg, avg_meters_benign['val_SE'].avg,
             avg_meters_benign['val_PC'].avg, avg_meters_benign['val_F1'].avg,
             avg_meters_benign['val_ACC'].avg))

    print('Malignant Test_loss %.4f - Malignant Test_iou %.4f - Malignant Test_dice %.4f - Malignant Test_SE %.4f - '
          'Test_PC %.4f - Test_F1 %.4f - Test_ACC %.4f '
          % (avg_meters_malignant['val_loss'].avg, avg_meters_malignant['val_iou'].avg,
             avg_meters_malignant['val_dice'].avg, avg_meters_malignant['val_SE'].avg,
             avg_meters_malignant['val_PC'].avg, avg_meters_malignant['val_F1'].avg,
             avg_meters_malignant['val_ACC'].avg))

    return "Evaluation Finished!"


if __name__ == "__main__":
    main(args)
