# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from models.architectures.models import Classification, Triplet
from utils.datasets import MarketDuke, RandomIdentitySampler
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation import eval_performance, train, save_model, load_model
from utils.losses import SoftRankingLoss
import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='PyTorch Re-id Training')
parser.add_argument('dataset_folder', type=str, help='Folder where the datasets are')
parser.add_argument('--dataset', default='market', type=str, help='dataset for training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--gpu', default=0, type=int, help='gpu id for cuda')
parser.add_argument('--n_epoch', '-n', default=20, type=int, help='number of epoch for training')
parser.add_argument('--triplet', action='store_true', help='if the training is done with triplet loss (either `triplet` or `classif` must be selected)')
parser.add_argument('--soft_margin', action='store_true', help='use the soft margin version of the triplet loss')
parser.add_argument('--classif', action='store_true', help='using classif model (either `triplet` or `classif` must be selected)')
parser.add_argument('--adv', action='store_true', help='GOAT (default: intra-batch, SMA: add flag sma, FNA: add flag push and/or pull)')
parser.add_argument('--push', action='store_true', help='Use pushing guides with GOAT')
parser.add_argument('--pull', action='store_true', help='Use pulling guide with GOAT')
parser.add_argument('--sma', action='store_true', help='use sma for adversarial training (no pulling/pushing guides)')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay for the optimizer')
parser.add_argument('--momentum', '-m', default=0.9, type=float, help='momentum for sgd')
parser.add_argument('--filename', '-f', help='filename of the saved model')
parser.add_argument('--pretrained', action='store_true', help='use the pretrained model or not')
parser.add_argument('--id_batch', action='store_true', help='use a random identity sampler')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='size of the batch size')
parser.add_argument('--num_instances', '-ni', default=4, type=int, help='number of instances for each identities in a batch, useful only for triplet. it must divide batch_size')
parser.add_argument('--embedding_size', '-e', default=2048, type=int, help='dimension of embedding (final layer output size)')
parser.add_argument('--resume', action='store_true', help='loading a checkpoint or not')
parser.add_argument('--comment', default='', help='comments for the tensorboard')
args = parser.parse_args()

assert args.triplet != args.classif, 'Either `triplet` or `classif` must be selected !'

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

train_transforms = transforms.Compose([
    # transforms.Resize((288,144)),
    transforms.Resize((256,128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    # RandomErasing()
    ])

test_transforms = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor()
        ])

if args.push:
    sorted_dataset = datasets.ImageFolder(os.path.join(args.dataset_folder, 'sorted', 'train'), transform=train_transforms)

    idx = defaultdict(list)
    for image,label in sorted_dataset:
        idx[label].append(image)

else:
    idx = None

train_dataset = MarketDuke(args.dataset_folder, name=args.dataset, set='train', transform=train_transforms)

if args.id_batch:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4,
                                            sampler=RandomIdentitySampler(train_dataset, args.num_instances, dataset=args.dataset),
                                            pin_memory=True, drop_last=True)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)


gallery_dataset = MarketDuke(args.dataset_folder, name=args.dataset, set='gallery',
                             transform=test_transforms)

probe_dataset = MarketDuke(args.dataset_folder, name=args.dataset, set='probe',
                           transform=test_transforms)

gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4, pin_memory=True)

probe_loader = torch.utils.data.DataLoader(probe_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4, pin_memory=True)


if args.classif:
    if args.dataset == 'market':
        num_classes = 751
    elif args.dataset == 'duke':
        num_classes = 702
    model = Classification(num_classes, pretrained=args.pretrained)
else:
    model = Triplet(args.embedding_size, pretrained=args.pretrained)

if args.triplet:
    if args.soft_margin:
        criterion = SoftRankingLoss(reduction='mean')
    else:
        criterion = nn.MarginRankingLoss(margin=10, reduction='mean')
else:
    criterion = nn.CrossEntropyLoss()

def lambdaf(epoch):
    if epoch > 50:
        return (0.001 ** ((epoch - 50)/100))
    else:
        return 1

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.classif:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
else:
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambdaf)

if args.resume and os.path.exists(os.path.join("models", "checkpoints", args.filename)):
    model, optimizer, best_map, start_epoch = load_model(model, os.path.join("models", "checkpoints"), args.filename, device)
    print('Model loaded')
else:
    start_epoch = 0
    best_map = 0
    print('New model')
    
model = model.to(device)
writer = SummaryWriter(comment=args.comment)

for epoch in range(start_epoch, start_epoch+args.n_epoch):
    print("\nEpoch {}".format(epoch))
    model, train_loss = train(model, criterion, optimizer, train_loader, device, writer, epoch, triplet=args.triplet, classif=args.classif, adv_training=args.adv, sma=args.sma, pushing_guides=idx, pull=args.pull, transforms=False)
    scheduler.step()
    writer.add_scalar("Training loss", train_loss, epoch)
    r1, r5, r10, MAP, gallery_features, gallery_ids = eval_performance(model, gallery_loader, probe_loader, device, triplet=args.triplet, classif=args.classif, cosine=False, transforms=True)
    print("map : {}, r1 : {}, r5 : {}, r10 : {}".format(MAP, r1, r5, r10))
    writer.add_scalar("Eval mAP", 100*MAP, epoch)
    writer.add_scalar("Eval Rank-1", 100*r1, epoch)
    best_map = save_model(MAP, best_map, model, optimizer, epoch, os.path.join("models", "checkpoints"), args.filename)