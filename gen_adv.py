# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch
from torchvision import transforms
from models.architectures.models import Classification, Triplet
from utils.datasets import MarketDuke
from utils.evaluation import load_model, extract_feature_img, mse
from utils.adv_evaluation import eval_performance_adv_query, eval_performance_adv_gal_cross_query, eval_performance_adv_query_cross_gal, eval_performance_adv_gal, odfa, max_min_mse, min_mse, multi_mse
from tqdm import tqdm
from collections import defaultdict
import advertorch.attacks
import numpy as np
import argparse
from time import time

parser = argparse.ArgumentParser(description='PyTorch Re-id Evaluating')
parser.add_argument('checkpoints_folder', help='Folder that contains the saved models')
parser.add_argument('dataset_folder', type=str, help='Folder where the datasets are')
parser.add_argument('filename', type=str, help='filename of the saved model')
parser.add_argument('--dataset', default='market', type=str, help='dataset for training')
parser.add_argument('--gpu', default=0, type=int, help='gpu id for cuda')
parser.add_argument('--triplet', action='store_true', help='if the training is done with triplet loss')
parser.add_argument('--embedding_size', '-e', default=2048, type=int, help='dimension of embedding (final layer output size)')
parser.add_argument('--cosine', action='store_true', help='use cosine similarity metric')
parser.add_argument('--classif', action='store_true', help='using classif model')
parser.add_argument('--transform', action='store_true', help='transform data before feature extraction')
parser.add_argument('--max_iter', default=15, type=int, help='number of iteration for iterative attacks')
parser.add_argument('--eps', default=5., type=float, help='maximum size of the perturbation')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='size of the batch size')
parser.add_argument('--attack', '-a', default='IFGSM', type=str, help='attack to use')
parser.add_argument('--single', action='store_true', help='use single shot attack')
parser.add_argument('--gallery', action='store_true', help='use gallery to attack')
parser.add_argument('--nb_repeat', '-n', default=1, type=int, help='number of repeat of the evaluation')
parser.add_argument('--cross_attack', action='store_true', help='use cross sets attack')
args = parser.parse_args()

assert args.triplet != args.classif, 'Either `triplet` or `classif` must be selected !'

print("========================== Parameters of evaluation ==========================")
for arg in vars(args):
    print(arg, getattr(args, arg))

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

args.clean = not args.transform

if args.clean:
    test_transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor()])
else:
    test_transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

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
    model = Classification(num_classes)
else:
    model = Triplet(args.embedding_size)

model = model.to(device)
model, _, best_map, epoch = load_model(model, None, args.checkpoints_folder, args.filename, device)


def wrapper(x):
    return extract_feature_img(model, x, cosine=args.cosine, triplet=args.triplet, classif=args.classif, transforms=args.clean)

if args.nb_repeat > 1:
    leave = False
else:
    leave = True


print(f"========================== epsilon = {args.eps} ==========================")

results = defaultdict(list)
for i in tqdm(range(args.nb_repeat)):

    epsilon = args.eps/255.
    # max_iter= int(min(e+4, 1.25*e))
    max_iter = args.max_iter
    self = False
    hardest = False
    if args.attack == 'IFGSM':
        if args.single:
            attack = advertorch.attacks.LinfPGDAttack(wrapper, mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=False)
        else:
            attack = advertorch.attacks.LinfPGDAttack(wrapper, multi_mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=False)
    elif args.attack == 'MIFGSM':
        if args.single:
            attack = advertorch.attacks.MomentumIterativeAttack(wrapper, mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, decay_factor=1.)
        else:
            attack = advertorch.attacks.MomentumIterativeAttack(wrapper, multi_mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, decay_factor=1.)
    elif args.attack == 'SMA':
        attack = advertorch.attacks.LinfPGDAttack(wrapper, mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=True)
        self = True
    elif args.attack == 'ODFA':
        if args.cosine:
            attack = advertorch.attacks.LinfPGDAttack(wrapper, odfa, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=False)
        else:
            attack = advertorch.attacks.LinfPGDAttack(wrapper, odfa, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=False)
        self = True
    elif args.attack == 'FNA':
        attack = advertorch.attacks.LinfPGDAttack(wrapper, max_min_mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=False)
        hardest = True
    elif args.attack == 'FNA_pull':
        attack = advertorch.attacks.LinfPGDAttack(wrapper, min_mse, eps=epsilon, nb_iter=max_iter, eps_iter=1.0/255.0, targeted=False, rand_init=False)
        hardest = True
    elif args.attack == 'FGSM':
        if args.single:
            attack = advertorch.attacks.GradientSignAttack(wrapper, mse, eps=epsilon, targeted=False)
        else:
            attack = advertorch.attacks.GradientSignAttack(wrapper, multi_mse, eps=epsilon, targeted=False)
    else:
        print("Unknown attack")

    t0 = time()
    if args.gallery:
        if args.cross_attack:
            r1, r5, r10, MAP = eval_performance_adv_gal_cross_query(model, gallery_loader, probe_loader, attack, device, cosine=args.cosine, triplet=args.triplet, classif=args.classif, single=args.single, transforms=args.clean, leave=leave)
        else:
            r1, r5, r10, MAP = eval_performance_adv_gal(model, gallery_loader, probe_loader, attack, device, cosine=args.cosine, triplet=args.triplet, classif=args.classif, transforms=args.clean, self=self, single=args.single, leave=leave)
    else:
        if args.cross_attack:
            r1, r5, r10, MAP = eval_performance_adv_query_cross_gal(model, gallery_loader, probe_loader, attack, device, cosine=args.cosine, triplet=args.triplet, classif=args.classif, single=args.single, transforms=args.clean, leave=leave)
        else:
            r1, r5, r10, MAP = eval_performance_adv_query(model, gallery_loader, probe_loader, attack, device, cosine=args.cosine, triplet=args.triplet, classif=args.classif, transforms=args.clean, self=self, hardest=hardest, single=args.single, leave=leave)
    t1 = time()
    results['r1'].append(r1)
    results['r5'].append(r5)
    results['r10'].append(r10)
    results['map'].append(MAP)
    results['time'].append(t1 - t0)

print(f"rank1 : mean = {np.mean(results['r1'])}, max = {np.max(results['r1'])}, min = {np.min(results['r1'])}, std = {np.std(results['r1'])}")
print(f"rank5 : mean = {np.mean(results['r5'])}, max = {np.max(results['r5'])}, min = {np.min(results['r5'])}, std = {np.std(results['r5'])}")
print(f"rank10 : mean = {np.mean(results['r10'])}, max = {np.max(results['r10'])}, min = {np.min(results['r10'])}, std = {np.std(results['r10'])}")
print(f"map : mean = {np.mean(results['map'])}, max = {np.max(results['map'])}, min = {np.min(results['map'])}, std = {np.std(results['map'])}")
print(f"Total computing time : {np.mean(results['time'])} sec")

print('===================================================================')