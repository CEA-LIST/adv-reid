# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch
import numpy as np
from tqdm import tqdm
from .losses import get_distances
import os
from torchvision.utils import make_grid
import advertorch.attacks
from collections import defaultdict

def extract_feature_img(model, data, cosine=False, triplet=True, classif=False, transforms=True, training=False):
    """Perform several transformations before extracting the feature of a batch of images.
    
    Arguments:
        model {Pytorch model} -- model used to extract the features
        data {Tensor} -- batch of images
    
    Keyword Arguments:
        cosine {bool} -- normalize the features for cosine similarity (default: {False})
        triplet {bool} -- model trained with triplet loss (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        transforms {bool} -- transform the images before extracting (default: {True})
    
    Returns:
        ff {Tensor} -- features extracted from the batch
    """
    img = data
    if transforms:
        # Resize and  Normalize
        img = torch.nn.functional.interpolate(img, size=(256, 128), mode='bilinear', align_corners=False)
        img -= torch.cuda.FloatTensor([[[0.485]], [[0.456]], [[0.406]]])
        img /= torch.cuda.FloatTensor([[[0.229]], [[0.224]], [[0.225]]])
    
    ff = prediction(model, img, triplet, classif, training)
    if cosine:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff / fnorm
    return ff


def prediction(model, input, triplet=True, classif=False, training=False):
    """Extract the features of a batch of images.
    
    Arguments:
        model {Pytorch model} -- model used to extract the features
        input {Tensor} -- batch of images
    
    Keyword Arguments:
        triplet {bool} -- model trained with triplet loss (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
    
    Returns:
        output {Tensor} -- features extracted from the batch
    """
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if not triplet and not classif:
        model.avgpool.register_forward_hook(get_activation('backbone'))
    
    if classif:
        output = model(input,training)
    elif not triplet:
        output = activation['backbone'].view(activation['backbone'].size(0), -1)
    else:
        output = model(input)
    
    return output


def extract_features(loader, model, device, cosine=False, triplet=True, classif=False, transforms=True, leave=True):
    """Extract all the features of images from a loader using a model.
    
    Arguments:
        loader {Pytorch dataloader} -- loader of the images
        model {Pytorch model} -- model used to extract the features
        device {cuda device} -- 
    
    Keyword Arguments:
        cosine {bool} -- Use cosine similarity (default: {False})
        triplet {bool} -- model trained with triplet loss (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        transforms {bool} -- perform transformation to the image (default: {True})
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """

    ids = []
    cams = []
    features = []
    model.eval()

    for _, data in tqdm(enumerate(loader), desc="Extracting features", total=len(loader), leave=leave):
        with torch.no_grad():
            output = extract_feature_img(model, data['image'].to(device), cosine=cosine, triplet=triplet, classif=classif, transforms=transforms)

        features.append(output)
        ids.append(data['id'].cpu())
        cams.append(data['cam'].cpu())

    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)

    return features.cpu(), ids.numpy(), cams.numpy()

def pairwise_distance(x, y):
    """Compute the matrix of pairwise distances between tensors x and y

    Args:
        x (Tensor)
        y (Tensor)

    Returns:
        Tensor: matrix of pairwise distances
    """
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None,
                 probe_views=None, ignore_MAP=True):
    """Compute the CMC and mAP for evaluation

    Args:
        dist (2D array): distance matrix of shape (num_gallery,num_probe)
        gallery_labels (1D array): array of gallery labels
        probe_labels (1D array): array of probe labels
        gallery_views (1D array, optional): if specified, for any probe image,
        the gallery correct matches from the same view are ignored. Defaults to None.
        probe_views (1D array, optional): must be specified if gallery_views are specified. Defaults to None.
        ignore_MAP (bool, optional): if true, only computes CMC. Defaults to True.

    Returns:
        1D array: CMC (in percent)
        int: mAP (in percent)
    """
    
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist.cpu())

    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        gallery_views = np.asarray(gallery_views)
        probe_views = np.asarray(probe_views)
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    ap = np.zeros((num_probe,))
    pbar = tqdm(desc="Evaluating", total=num_probe)
    for i in range(num_probe):
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels

        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        cmc[:, i] = cmc_

        if not ignore_MAP:
            num_correct = positions_correct.shape[0]
            for j in range(num_correct):
                last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0
                current_precision = float(j + 1) / float(positions_correct[j] + 1)
                ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)
        pbar.set_postfix({'AP': ap[i]})
        pbar.update()
    pbar.close()
    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(ap)
    return CMC * 100, MAP * 100

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

def evaluate(score, ql, qc, gl, gc, cosine=False):
    # predict index
    index = np.argsort(score)  # from small to large
    if cosine:
        index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
    return ap_tmp, CMC_tmp

def eval_performance(model, gallery_loader, probe_loader, device, triplet=True, classif=False, cosine=False, transforms=False):

    gallery_features, gallery_ids, gallery_cams = extract_features(gallery_loader, model, device, cosine, triplet, classif, transforms)
    probe_features, probe_ids, probe_cams = extract_features(probe_loader, model, device, cosine, triplet, classif, transforms)
    dist = pairwise_distance(probe_features, gallery_features)
    # CMC, MAP = eval_cmc_map(dist, gallery_ids, probe_ids, gallery_cams, probe_cams, ignore_MAP=False)
    CMC = torch.IntTensor(len(gallery_ids)).zero_()
    ap = 0.0
    for i, qf in tqdm(enumerate(dist.cpu().numpy()), total=len(dist)):
        ap_tmp, CMC_tmp = evaluate(qf, probe_ids[i], probe_cams[i], gallery_ids, gallery_cams)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(probe_ids)  # average CMC
    mAP = ap / len(probe_ids)
    r1 = CMC[0]
    r5 = CMC[4]
    r10 = CMC[9]

    return r1, r5, r10, mAP, gallery_features, gallery_ids


def get_triplet_from_batch(list_triplet, batch):
    images = []
    for triplet in list_triplet:
        triplet_images = []
        triplet_images.append(batch['image'][triplet[0]])
        triplet_images.append(batch['image'][triplet[1]])
        triplet_images.append(batch['image'][triplet[2]])
        triplet_images = torch.stack(triplet_images)
        images.append(triplet_images)
    return images

mse = torch.nn.MSELoss(reduction='sum')

def sum_mse(b,f):
    tmp = 0
    for x in f:
        tmp += mse(b,x)
    return tmp

def sum_dif_mse(b,f):
    tmp = 0
    for i in range(len(f)-1):
        tmp += mse(b,f[i])
    tmp -= mse(b,f[-1])
    return tmp

def create_adv_batch(model, inputs, labels, device, epsilon=5.0, momentum=1.0, rand_r=True, classif=False, triplet=True, request_dict=None, self_sufficient=False, transforms=True, nb_r=1, pull=False):
    """Create an adversarial batch from a given batch for adversarial training with GOAT

    Args:
        model (nn.Module): model that will serve to create the adversarial images
        inputs (Tensor): batch of original images
        labels (Tensor): labels of the original images
        device (cuda Device): cuda device
        epsilon (int, optional): adversarial perturbation budget. Defaults to 5.0.
        momentum (float, optional): momentum for the attack. Defaults to 1.0.
        rand_r (bool, optional): whether the guides are chosen at random. Defaults to True.
        classif (bool, optional): whether the models is trained by cross entropy or not. Defaults to False.
        triplet (bool, optional): whether the models is trained with triplet loss. Defaults to True.
        request_dict (dict, optional): dictionary that contains the possible guides. Defaults to None.
        self_sufficient (bool, optional): whether the attack is self sufficient (use an artificial guide). Defaults to False.
        transforms (bool, optional): whether to apply the transformation or not. Defaults to True.
        nb_r (int, optional): number of guides. Defaults to 1.
        pull (bool, optional): whether to sample pulling guides. Defaults to False.

    Returns:
        Tensor: batch of adversarial images
    """
    
    if request_dict: # GOAT : INTER BATCH
        requests = []
        if nb_r > 1:
            for i in range(nb_r):
                R = []
                for label in labels:
                    image = request_dict[label.cpu().numpy()[0]]
                    if rand_r:
                        r = np.random.randint(len(image))
                    else:
                        r = i%(len(image))
                    R.append(image[r])
                R = torch.stack(R).to(device)
                with torch.no_grad():
                    R = extract_feature_img(model, R, cosine=False, triplet=triplet, classif=classif, transforms=transforms)
                requests.append(R)
            requests = torch.stack(requests)
            criterion = sum_mse
        else:
            for label in labels:
                image = request_dict[label.cpu().numpy()[0]]
                if rand_r:
                    r = np.random.randint(len(image))
                else:
                    r = 0
                requests.append(image[r])
            requests = torch.stack(requests)
            criterion = mse
            with torch.no_grad():
                # requests = prediction(model, requests, triplet=triplet, classif=classif)
                requests = extract_feature_img(model, requests, cosine=False, triplet=triplet, classif=classif, transforms=transforms)
        if pull: # GOAT FNA : INTER BATCH
            # FURTHEST INTRA
            with torch.no_grad():
                features = extract_feature_img(model, inputs, cosine=False, triplet=triplet, classif=classif, transforms=transforms)
            dist_intra = pairwise_distance(features, features) # pairwise distance between features in batch
            pulling = []
            for nb_f in range(len(features)):
                dist_f = dist_intra[nb_f] # list of distances to feature nb_f
                # find max distance in batch
                max_d_p = -np.inf # max distance
                max_ind_p = 0 # max index
                for i,d_p in enumerate(dist_f):
                    if d_p > max_d_p and i != nb_f:
                        max_d_p = d_p 
                        max_ind_p = i
                pulling.append(features[max_ind_p])
            pulling = torch.stack(pulling)
            # add pulling feature at the end of pushing guides -> single pulling guide
            requests = torch.cat((requests,pulling.unsqueeze(0)))
            criterion = sum_dif_mse
        requests = requests.to(device)

    elif self_sufficient: # ONLINE WITH SMA
        criterion = mse
        with torch.no_grad():
            # requests = prediction(model, inputs, triplet=triplet, classif=classif)
            requests = extract_feature_img(model, inputs, cosine=False, triplet=triplet, classif=classif, transforms=transforms)
    else: # INTRA BATCH (TRIPLET ONLY)
        criterion = sum_mse
        with torch.no_grad():
            # features = prediction(model, inputs, triplet=triplet, classif=classif)
            features = extract_feature_img(model, inputs, cosine=False, triplet=triplet, classif=classif, transforms=transforms)
        index_dic = defaultdict(list)
        
        for i, target in enumerate(labels):
            index_dic[target.item()].append(i)
        if rand_r:
            requests = []
            index_adv = []
            for i, target in enumerate(labels):
                r = i
                while r == i:
                    r = np.random.randint(len(index_dic[target.item()]))
                x_r = index_dic[target.item()][r]
                requests.append(features[x_r])
                index_adv.append(x_r)
            requests = torch.stack(requests)
        else:
            requests = []
            for i in range(nb_r):
                R = []
                for target in labels:
                    R.append(features[index_dic[target.item()][i]])
                R = torch.stack(R)
                requests.append(R)
            requests = torch.stack(requests).to(device)
    
    max_iter = 7
    attack = advertorch.attacks.PGDAttack(lambda x: extract_feature_img(model, x, triplet=triplet, classif=classif, cosine=False, transforms=transforms), criterion, eps=epsilon/255.0, nb_iter=max_iter, eps_iter=1.0/255.0, ord=np.inf, clip_max=inputs.max(), clip_min=inputs.min(), rand_init=True)
    data_adv = attack.perturb(inputs, requests)
    return data_adv

def train(model, criterion, optimizer, dataloader, device, writer, epoch, triplet=False, classif=False, log_images=False, adv_training=False, pushing_guides=None, transforms=False, sma=False, pull=False):
    """Training loop of a re-ID model

    Args:
        model (nn.Module): model to train
        criterion (nn.Module): loss function to optimize
        optimizer (torch.optim): optimizer for the training
        dataloader (DataLoader): dataloader to loop on
        device (torch.device): cuda device of the tensors and model
        writer (SummaryWriter): writer for tensorboard
        epoch (int): current epoch
        triplet (bool, optional): whether the model use a triplet loss (either `triplet` or `classif` must be set to True). Defaults to False.
        classif (bool, optional): whether the model use a cross entropy loss (either `triplet` or `classif` must be set to True). Defaults to False.
        log_images (bool, optional): whether to save the first images in tensorboard for visualization. Defaults to False.
        adv_training (bool, optional): whether to use adversarial training or not (GOAT). Defaults to False.
        pushing_guides (dict, optional): dictionary of pushing guides to sample from. Defaults to None.
        transforms (bool, optional): whether to apply transforms on the images or not. Defaults to False.
        sma (bool, optional): whether to use a SMA for adversarial training. Defaults to False.
        pull (bool, optional): whether to sample pulling guides for GOAT. Defaults to False.

    Returns:
        nn.Module: trained model
        int: mean training loss for this epoch
    """

    running_loss, running_corrects = 0, 0
    total = 0
    model.train()
    pbar = tqdm(desc="Training", total=len(dataloader))
    for index, batch in enumerate(dataloader):
        if not adv_training:
            inputs = batch['image'].float().to(device)
            labels = batch['id'].to(device)
        elif adv_training:
            inputs_clean = batch['image'].float().to(device)
            labels = batch['id'].to(device)
            
            inputs = create_adv_batch(model, inputs_clean, labels, device, rand_r=True, nb_r=4, classif=classif, triplet=triplet, request_dict=pushing_guides, pull=pull, self_sufficient=sma, transforms=transforms)
        
        if index == 0 and log_images:
            grid = make_grid(inputs)
            writer.add_image("First batch", grid, epoch)

        # zero the parameter gradients
        optimizer.zero_grad()

        if triplet:
            outputs = extract_feature_img(model, inputs, cosine=False, triplet=triplet, classif=classif, transforms=transforms)
            dist = pairwise_distance(outputs, outputs)
            dist_ap, dist_an, list_triplet = get_distances(dist, labels)
            y = torch.ones(dist_ap.size(0)).to(device)
            loss = criterion(dist_an, dist_ap, y)
            if index == 0 and log_images:
                list_images = get_triplet_from_batch(list_triplet, batch)
                grid = make_grid(list_images[0])
                writer.add_image("Triplets first batch", grid, epoch)

        else: #if classif
            outputs = extract_feature_img(model, inputs, cosine=False, triplet=triplet, classif=classif, transforms=transforms, training=True)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.squeeze())

        loss.backward()
        optimizer.step()

        postfix = {}
        running_loss += loss.item()
        mean_loss = running_loss/(index+1)
        postfix['Loss'] = mean_loss
        if classif:
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            postfix['Correct'] = '{}/{}'.format(running_corrects, total)
        pbar.update()
        pbar.set_postfix(postfix)
    pbar.close()
    return model, mean_loss


def load_model(net, optim, save_path, filename, device):
    """Load a model and its optimizer

    Args:
        net (nn.Module): architecture of the saved model
        optim (torch.optim): optimizer to load
        save_path (str): path where the file is stored
        filename (str): filename to open
        device (torch.device): device to load the model and optimizer to

    Returns:
        nn.Module: loaded model
        torch.optim: optimizer of the loaded model
        int: performance of the model
        int: number of epoch the model were trained
    """
    state = torch.load(os.path.join(save_path, filename), map_location=device)
    net.load_state_dict(state['net'])
    best_map = state['map']
    epoch = state['epoch']

    try:
        optim_state = state['optim']
    except KeyError:
        optim_state = None
    
    if optim_state and optim:
        optim.load_state_dict(optim_state)

    return net, optim, best_map, epoch


def save_model(MAP, best_map, net, optim, epoch, save_path, filename):
    """Save a model and its optimizer if its mAP is better than the saved one

    Args:
        MAP (int): performance of the model to save
        best_map (int): saved model performance
        net (nn.Module): model to save
        optim (torch.optim): optimizer of the model to save
        epoch (int): number of epoch the model were trained
        save_path (str): path on disk where to save the model to
        filename (str): filename on disk

    Returns:
        int: the saved model performance
    """
    if MAP > best_map:
        print('Saving ...')
        state = {
            'net': net.state_dict(),
            'map': MAP,
            'epoch': epoch,
            'optim': optim.state_dict()
        }
        torch.save(state, os.path.join(save_path, filename))
        best_map = MAP
    return best_map