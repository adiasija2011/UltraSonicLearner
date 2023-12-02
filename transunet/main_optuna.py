import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import wandb
import optuna
from optuna.integration import WeightsAndBiasesCallback
from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score
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
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=300, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=41, help='random seed')
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
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                            transform=train_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4)

    return trainloader, valloader

def objective(trial):
    base_lr = trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    args.base_lr = base_lr
    args.batch_size = batch_size

    trainloader, valloader = getDataloader(args=args)
    model = get_model(args)
    model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    best_val_iou = 0
    for epoch_num in range(args.epoch):
    # Training loop
        model.train()
        avg_meters = {'loss': AverageMeter(),
                        'iou': AverageMeter()}

        for i_batch, sampled_batch in enumerate(trainloader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

            outputs = model(img_batch)
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))

        # Validation loop
        model.eval()
        val_avg_meters = {'val_loss': AverageMeter(),
                            'val_iou': AverageMeter()}
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

                output = model(img_batch)
                loss = criterion(output, label_batch)
                iou, _, SE, PC, F1, _, ACC = iou_score(output, label_batch)

                val_avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                val_avg_meters['val_iou'].update(iou, img_batch.size(0))

        # Check if the current model is the best one
        if val_avg_meters['val_iou'].avg > best_val_iou:
            best_val_iou = val_avg_meters['val_iou'].avg

    return best_val_iou

def main(args):
    wandb.init(project='ultrasoniclearners', entity='adiasija10')
    wandb.config.update(args)

    wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": "ultrasoniclearners-optuna"})
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, callbacks=[wandbc])

    wandb.finish()

if __name__ == "__main__":
    main(args)
