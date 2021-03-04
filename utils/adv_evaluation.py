# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .evaluation import extract_feature_img, pairwise_distance, evaluate, eval_cmc_map, extract_features, mse
from time import time
from torchvision import transforms

toImage = transforms.ToPILImage()

def extract_features_adv(g_loader, q_loader, model, attack, device, triplet=True, classif=False, cosine=False, transforms=False, single=True, leave=True):
    """Perturb the images from g_loader using the images from q_loader as guides
    
    Arguments:
        g_loader {Pytorch Dataloader} -- gallery loader (images to attack)
        q_loader {Pytorch Dataloader} -- query loader (guides for the attack)
        model {Pytorch model} -- model used the extract the features
        attack {advertorch.attack} -- attack used
        device {torch device}
            
    Keyword Arguments:
        triplet {bool} -- model trained with triplet loss (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        cosine {bool} -- whether to use cosine similarity as the metric
        transforms {bool} -- whether to apply the transforms to the features
        single {bool} -- whether to use a single guide
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
        
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()
                
    dict_q_features = defaultdict(list)
    for data in q_loader:
        with torch.no_grad():
            f_data = extract_feature_img(model, data['image'].to(device), cosine=cosine, triplet=triplet, classif=classif, transforms=transforms)
        for feature, id in zip(f_data, data['id']):
            dict_q_features[id.cpu().item()].append(feature)
    
    dict_g_id = defaultdict(list)
    distract_images = []
    for data in g_loader:
        for image, id, cam in zip(torch.unbind(data['image']), data['id'], data['cam']):
            if id.cpu().item() == 0 or id.cpu().item() == -1:
                distract_images.append(image)
                cams.append(cam)
                ids.append(id)
            else:
                dict_g_id[id.cpu().item()].append((image, cam))
    if distract_images:
        distract_images = torch.stack(distract_images)
        for distract in torch.split(distract_images, 64):
            with torch.no_grad():
                distract_features = extract_feature_img(model, distract.to(device), cosine=cosine, triplet=triplet, classif=classif, transforms=transforms)
            features.append(distract_features)
    for id, images in tqdm(dict_g_id.items(), leave=leave):
        tmp_gallery = []
        tmp_queries = []
        for elt in images:
            tmp_gallery.append(elt[0])
            cams.append(elt[1])
            ids.append(torch.tensor(id))
        if single:
            for _ in range(len(tmp_gallery)):
                r = np.random.randint(len(dict_q_features[id]))
                tmp_queries.append(dict_q_features[id][r])

        tmp_gallery = torch.stack(tmp_gallery)
        if single:
            tmp_queries = torch.stack(tmp_queries)
        else:
            tmp_queries = torch.stack(dict_q_features[id]).repeat((tmp_gallery.shape[0], 1, 1)).transpose(0,1)
        epsilon = attack.eps
        attack.clip_max = tmp_gallery.max()
        attack.clip_min = tmp_gallery.min()

        image_adv = attack.perturb(tmp_gallery.to(device), tmp_queries.to(device))
        for batch in torch.split(image_adv, 64):
            with torch.no_grad():
                output_adv = extract_feature_img(model, batch.to(device), cosine=cosine, triplet=triplet, classif=classif, transforms=transforms)
            features.append(output_adv)

    ids = torch.stack(ids)
    cams = torch.stack(cams)
    features = torch.cat(features, 0)
    

    return features.cpu(), ids.numpy(), cams.numpy()

def eval_performance_adv_gal_cross_query(model, gallery_loader, probe_loader, attack, device, triplet=True, classif=False, transforms=False, cosine=False, single=True, leave=True):
    """Evaluate the performance when the gallery is attacked using guides from the query.
    
    Arguments:
        model {Pytorch model} -- model to evaluate
        gallery_loader {Pytorch dataloader} -- gallery loader
        probe_loader {Pytorch dataloader} -- query loader
        attack {advertorch.attack} -- attack used
        device {torch device}
    
    Keyword Arguments:
        triplet {bool} -- model trained with triplet loss (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        cosine {bool} -- whether to use cosine similarity as the metric
        transforms {bool} -- whether to apply the transforms to the features
        single {bool} -- whether to use a single guide
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        r1 -- rank1 accuracy
        r5 -- rank5 accuracy
        r10 -- rank10 accuracy
        mAP -- mAP of the model
    """

    gallery_features, gallery_ids, gallery_cams = extract_features_adv(gallery_loader, probe_loader, model, attack, device, triplet=triplet, classif=classif, transforms=transforms, cosine=cosine, single=single, leave=leave)
    probe_features, probe_ids, probe_cams = extract_features(probe_loader, model, device, cosine=cosine, triplet=triplet, classif=classif, transforms=transforms, leave=leave)
    if not cosine:
        dist = pairwise_distance(probe_features, gallery_features)
    # CMC, MAP = eval_cmc_map(dist, gallery_ids, probe_ids, gallery_cams, probe_cams, ignore_MAP=False)
    CMC = torch.IntTensor(len(gallery_ids)).zero_()
    ap = 0.0
    for i in tqdm(range(len(probe_ids)), leave=leave):
        if not cosine:
            score = np.asarray(dist[i].cpu())
        else:
            qf = np.asarray(probe_features[i])
            score = np.dot(gallery_features, qf)
        ap_tmp, CMC_tmp = evaluate(score, probe_ids[i], probe_cams[i], gallery_ids, gallery_cams, cosine)
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

    return r1, r5, r10, mAP

def eval_performance_adv_query_cross_gal(model, gallery_loader, probe_loader, attack, device, triplet=True, classif=False, transforms=False, cosine=False, single=True, leave=True):
    """Evaluate the performance when the query is attacked using guides from the gallery.
    
    Arguments:
        model {Pytorch model} -- model to evaluate
        gallery_loader {Pytorch dataloader} -- gallery loader
        probe_loader {Pytorch dataloader} -- query loader
        attack {advertorch.attack} -- attack used
        device {torch device}
    
    Keyword Arguments:
        triplet {bool} -- model trained with triplet loss (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        cosine {bool} -- whether to use cosine similarity as the metric
        transforms {bool} -- whether to apply the transforms to the features
        single {bool} -- whether to use a single guide
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        r1 -- rank1 accuracy
        r5 -- rank5 accuracy
        r10 -- rank10 accuracy
        mAP -- mAP of the model
    """

    gallery_features, gallery_ids, gallery_cams = extract_features(gallery_loader, model, device, cosine=cosine, triplet=triplet, classif=classif, transforms=transforms, leave=leave)
    probe_features, probe_ids, probe_cams = extract_features_adv(probe_loader, gallery_loader, model, attack, device, cosine=cosine, triplet=triplet, classif=classif, transforms=transforms, single=single, leave=leave)
    if not cosine:
        dist = pairwise_distance(probe_features, gallery_features)
    # CMC, MAP = eval_cmc_map(dist, gallery_ids, probe_ids, gallery_cams, probe_cams, ignore_MAP=False)
    CMC = torch.IntTensor(len(gallery_ids)).zero_()
    ap = 0.0
    for i in tqdm(range(len(probe_ids)), leave=leave):
        if not cosine:
            score = np.asarray(dist[i].cpu())
        else:
            qf = np.asarray(probe_features[i])
            score = np.dot(gallery_features, qf)
        ap_tmp, CMC_tmp = evaluate(score, probe_ids[i], probe_cams[i], gallery_ids, gallery_cams, cosine)
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

    return r1, r5, r10, mAP

def perturb_queries(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True):
    """Perturb the queries during evaluation with multiple guides
    
    Arguments:
        q_loader {pytorch dataloader} -- dataloader of the query dataset
        attack {advertorch.attack} -- adversarial attack to perform on the queries
        model {pytorch model} -- pytorch model to evaluate
        device {torch device} --
        cosine {bool} -- evaluate with cosine similarity
        triplet {bool} -- model trained with triplet or not
        classif {bool} -- model trained with cross entropy
        transforms {bool} -- whether to apply the transforms to the features
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()

    dict_q_features = defaultdict(list)
    raw_features = []
    raw_ids = []
    guide_features = []
    for data in tqdm(q_loader, desc='Creating dict', leave=leave):
        with torch.no_grad():
            f_data = extract_feature_img(model, data['image'].to(device), cosine, triplet, classif, transforms)
        for feature, image, id, cam in zip(f_data, data['image'], data['id'].cpu(), data['cam'].cpu()):
            dict_q_features[id.item()].append(feature)
            raw_features.append(feature)
            raw_ids.append(id.item())
            ids.append(id)
            cams.append(cam)

    max_n_id = 0
    for _, values in dict_q_features.items():
        if len(values) > max_n_id:
            max_n_id = len(values)

    for f_x, id in tqdm(zip(raw_features, raw_ids), total=len(raw_features)):
        guide = []
        for f in dict_q_features[id]:
            guide.append(f)
        while len(guide) < max_n_id:
            guide.append(f_x)
        guide_features.append(torch.stack(guide))

    guide_features = torch.stack(guide_features)
    b_s = q_loader.batch_size

    for guides, data in tqdm(zip(torch.split(guide_features, b_s), q_loader), total=len(q_loader)):
        epsilon = attack.eps
        attack.clip_max = data['image'].max()
        attack.clip_min = data['image'].min()
        # attack.loss_fn = lambda x,y: n_f*nored_mse(x,y)
        image_adv = attack.perturb(data['image'].to(device), guides.to(device))
        with torch.no_grad():
            output = extract_feature_img(model, image_adv.to(device), cosine, triplet, classif, transforms)
        features.append(output)
        # ids.append(data['id'].cpu())
        # cams.append(data['cam'].cpu())

    ids = torch.stack(ids)
    cams = torch.stack(cams)
    features = torch.cat(features, 0)

    return features.cpu(), ids.numpy(), cams.numpy()

def valid_perm(p):
    """Find if a permutation is 'valid' i.e if all the inputs are shuffled
    
    Arguments:
        p {List} -- Permutation of indices
    
    Returns:
        Bool -- True if the permutation is valid, False otherwise
    """
    for i, pi in enumerate(p):
        if i == pi:
            return False
    return True

def perturb_queries_single(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True):
    """Perturb the queries during evaluation with a single guide
    
    Arguments:
        q_loader {pytorch dataloader} -- dataloader of the query dataset
        attack {advertorch.attack} -- adversarial attack to perform on the queries
        model {pytorch model} -- pytorch model to evaluate
        device {cuda device} --
        cosine {bool} -- evaluate with cosine similarity
        triplet {bool} -- model trained with triplet or not
        classif {bool} -- model trained with cross entropy
        transforms {bool} -- whether to apply the transforms to the features
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()
    T = []
    
    # Creating a dictionary mapping key=id, value=list of (feature, image, cam, id) of the queries for this id 
    dict_q_features = defaultdict(list)
    for data in tqdm(q_loader, desc='Creating dict', leave=leave):
        with torch.no_grad():
            f_data = extract_feature_img(model, data['image'].to(device), cosine, triplet, classif, transforms)
        for feature, image, id, cam in zip(f_data, data['image'], data['id'].cpu(), data['cam'].cpu()):
            dict_q_features[id.item()].append((feature, image, cam, id))
    
    for id_dict, queries in tqdm(dict_q_features.items(), leave=leave):
        tmp_queries = []
        raw_features = []
        for elt in queries:
            raw_features.append(elt[0])
            tmp_queries.append(elt[1])
            cams.append(elt[2])
            ids.append(elt[3])
        tmp_queries = torch.stack(tmp_queries)
        raw_features = torch.stack(raw_features)
        if id_dict != 0 and id_dict != -1:
            epsilon = attack.eps
            attack.clip_max = tmp_queries.max()
            attack.clip_min = tmp_queries.min()

            # raw_features is the features of tmp_queries with the same indices
            # to take a single random query, we shuffle the features so that every feature is at a different place
            p = torch.randperm(raw_features.size(0))
            while not valid_perm(p):
                p = torch.randperm(raw_features.size(0))
            raw_features = raw_features[p].to(device)
            t0 = time()
            image_adv = attack.perturb(tmp_queries.to(device), raw_features.to(device))
            t1 = time() - t0
            T.append(t1)
            with torch.no_grad():
                output = extract_feature_img(model, image_adv.to(device), cosine, triplet, classif, transforms)
        else:
            output = raw_features

        features.append(output)

    ids = torch.stack(ids)
    cams = torch.stack(cams)
    features = torch.cat(features, 0)
    # print(f"\ntotal time : {sum(T)}, mean time : {np.mean(T)}" )

    return features.cpu(), ids.numpy(), cams.numpy()

#############################################################

def perturb_queries_naive(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True):
    ids = []
    cams = []
    features = []
    model.eval()

    # Creating a dictionary mapping key=id, value=list of (feature, image, cam, id) of the queries for this id 
    dict_q_features = defaultdict(list)
    dict_q_images = defaultdict(list)
    for data in tqdm(q_loader, desc='Creating dict'):
        with torch.no_grad():
            f_data = extract_feature_img(model, data['image'].to(device), cosine, triplet, classif, transforms)
        for feature, image, id in zip(f_data, data['image'], data['id'].cpu()):
            dict_q_features[id.item()].append(feature)
            dict_q_images[id.item()].append(image)

    for data in tqdm(q_loader):
        raw_features = []
        for image,label in zip(data['image'],data['id'].cpu()):
            r = np.random.randint(len(dict_q_features[label.item()]))
            while torch.equal(dict_q_images[label.item()][r], image):
                # print("not good")
                r = np.random.randint(len(dict_q_features[label.item()])) 
            raw_features.append(dict_q_features[label.item()][r])
        raw_features = torch.stack(raw_features).to(device)
        epsilon = attack.eps
        attack.clip_max = data['image'].max()
        attack.clip_min = data['image'].min()
        image_adv = attack.perturb(data['image'].to(device), raw_features.to(device))
        with torch.no_grad():
            output = extract_feature_img(model, image_adv.to(device), cosine, triplet, classif, transforms)
        features.append(output)
        ids.append(data['id'].cpu())
        cams.append(data['cam'].cpu())
        
    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)

    return features.cpu(), ids.numpy(), cams.numpy()

##################################################################


def perturb_queries_self(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True):
    """Perturb the queries with SMA
    
    Arguments:
        q_loader {pytorch dataloader} -- dataloader of the query dataset
        attack {advertorch.attack} -- adversarial attack to perform on the queries
        model {pytorch model} -- pytorch model to evaluate
        device {cuda device} --
        cosine {bool} -- evaluate with cosine similarity
        triplet {bool} -- model trained with triplet or not
        classif {bool} -- model trained with cross entropy
        transforms {bool} -- apply transforms
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()
    T = []

    for data in tqdm(q_loader, desc="Extracting features", leave=leave):
        with torch.no_grad():
            raw_features = extract_feature_img(model, data['image'].to(device), cosine, triplet, classif, transforms)
        
        epsilon = attack.eps
        attack.clip_max = data['image'].max()
        attack.clip_min = data['image'].min()
        image_adv = attack.perturb(data['image'].to(device), raw_features.to(device))
        with torch.no_grad():
            output = extract_feature_img(model, image_adv.to(device), cosine, triplet, classif, transforms)
        features.append(output)
        ids.append(data['id'].cpu())
        cams.append(data['cam'].cpu())

    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)

    return features.cpu(), ids.numpy(), cams.numpy()

def eval_performance_adv_query(model, gallery_loader, probe_loader, attack, device, cosine, triplet=True, classif=False, transforms=True, self=False, single=False, hardest=False, rand=False, leave=True):
    """Evaluate performance when the query is attacked with images from the query
    
    Arguments:
        model {pytorch model} -- model to evaluate
        gallery_loader {pytorch dataloader} -- dataloader of the gallery
        probe_loader {pytorch dataloader} -- dataloader of the query
        attack {advertorch.attack} -- attack used to perturb the queries
        device {cuda device} --
        cosine {bool} -- use cosine similarity or not
    
    Keyword Arguments:
        triplet {bool} -- model trained with triplet (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        transforms {bool} -- transform before attack (default: {True})
        self {bool} -- perturb the queries using themselves (default: {False})
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        r1 -- rank1 accuracy
        r5 -- rank5 accuracy
        r10 -- rank10 accuracy
        mAP -- mAP of the model
    """
    gallery_features, gallery_ids, gallery_cams = extract_features(gallery_loader, model, device, cosine, triplet, classif, transforms, leave=leave)
    if self:
        probe_features, probe_ids, probe_cams = perturb_queries_self(probe_loader, attack, model, device, cosine, triplet, classif, transforms, leave=leave)
    elif hardest:
        if rand:
            probe_features, probe_ids, probe_cams = perturb_queries_hardest_multi(probe_loader, attack, model, device, cosine, triplet, classif, transforms, leave, rand=True)
        else:
            probe_features, probe_ids, probe_cams = perturb_queries_hardest_multi(probe_loader, attack, model, device, cosine, triplet, classif, transforms, leave)
    else:
        if single:
            probe_features, probe_ids, probe_cams = perturb_queries_naive(probe_loader, attack, model, device, cosine, triplet, classif, transforms, leave)
            # probe_features, probe_ids, probe_cams = perturb_queries_single(probe_loader, attack, model, device, cosine, triplet, classif, transforms, leave=leave)
        else:
            probe_features, probe_ids, probe_cams = perturb_queries(probe_loader, attack, model, device, cosine, triplet, classif, transforms, leave=leave)
    if not cosine:
        dist = pairwise_distance(probe_features, gallery_features)
    # CMC, MAP = eval_cmc_map(dist, gallery_ids, probe_ids, gallery_cams, probe_cams, ignore_MAP=False)
    CMC = torch.IntTensor(len(gallery_ids)).zero_()
    ap = 0.0
    for i in tqdm(range(len(probe_ids)), leave=leave):
        if not cosine:
            score = np.asarray(dist[i].cpu())
        else:
            qf = np.asarray(probe_features[i])
            score = np.dot(gallery_features, qf)
        ap_tmp, CMC_tmp = evaluate(score, probe_ids[i], probe_cams[i], gallery_ids, gallery_cams, cosine)
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

    return r1, r5, r10, mAP

def eval_performance_adv_gal(model, gallery_loader, probe_loader, attack, device, cosine, triplet=True, classif=False, transforms=True, self=False, single=False, leave=True):
    """Evaluate performance when the gallery is attacked with images from the gallery
    
    Arguments:
        model {pytorch model} -- model to evaluate
        gallery_loader {pytorch dataloader} -- dataloader of the gallery
        probe_loader {pytorch dataloader} -- dataloader of the query
        attack {advertorch.attack} -- attack used to perturb the queries
        device {cuda device} --
        cosine {bool} -- use cosine similarity or not
    
    Keyword Arguments:
        triplet {bool} -- model trained with triplet (default: {True})
        classif {bool} -- model trained with cross entropy (default: {False})
        transforms {bool} -- transform before attack (default: {True})
        self {bool} -- perturb the queries using themselves (default: {False})
        leave {bool} -- whether to leave progress bar
                        (useful if the evaluation is repeated multiple times)
    
    Returns:
        r1 -- rank1 accuracy
        r5 -- rank5 accuracy
        r10 -- rank10 accuracy
        mAP -- mAP of the model
    """
    probe_features, probe_ids, probe_cams = extract_features(probe_loader, model, device, cosine, triplet, classif, transforms, leave=leave)
    if self:
        gallery_features, gallery_ids, gallery_cams = perturb_queries_self(gallery_loader, attack, model, device, cosine, triplet, classif, transforms, leave=leave)
    elif single:
        gallery_features, gallery_ids, gallery_cams = perturb_queries_single(gallery_loader, attack, model, device, cosine, triplet, classif, transforms, leave=leave)
    else:
        gallery_features, gallery_ids, gallery_cams = perturb_queries(gallery_loader, attack, model, device, cosine, triplet, classif, transforms, leave=leave)
    if not cosine:
        dist = pairwise_distance(probe_features, gallery_features)
    # CMC, MAP = eval_cmc_map(dist, gallery_ids, probe_ids, gallery_cams, probe_cams, ignore_MAP=False)
    CMC = torch.IntTensor(len(gallery_ids)).zero_()
    ap = 0.0
    for i in tqdm(range(len(probe_ids)), leave=leave):
        if not cosine:
            score = np.asarray(dist[i].cpu())
        else:
            qf = np.asarray(probe_features[i])
            score = np.dot(gallery_features, qf)
        ap_tmp, CMC_tmp = evaluate(score, probe_ids[i], probe_cams[i], gallery_ids, gallery_cams, cosine)
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

    return r1, r5, r10, mAP

def double_sum_mse(f1s, f2s):
    ret = 0
    for f1 in f1s:
        for f2 in f2s:
            ret += mse(f1, f2)
    return ret

def multi_mse(f1s, f2s):
    m = 0
    for f1, f2 in zip(f1s, f2s):
        for f in f2:
            m += mse(f1,f)
    return m

def max_min_mse(f1s, f2s):
    # f1, f2 = fs[:,0,:], fs[:,1,:]
    # return mse(x,f1) - mse(x,f2)
    m = 0
    for f1, f2 in zip(f1s, f2s):
        for i in range(len(f2)-1):
            m += mse(f1,f2[i])
        m -= mse(f1,f2[-1])
    return m

def min_mse(f1s, f2s):
    m = 0
    for f1, f2 in zip(f1s, f2s):
        m -= mse(f1,f2[-1])
    return m

def odfa(f1, f2):
    return mse(f1, -f2)

def odfa_sum(f1s, f2s):
    ret = 0
    for f1 in f1s:
        for f2 in f2s:
            ret += mse(f1, -f2)
    return ret

def cosine_loss(f1, f2):
    ff1 = torch.norm(f1, p=2)
    ff2 = torch.norm(f2, p=2)
    f1 = f1 / ff1
    f2 = f2 / ff2

    ff = np.dot(f1.cpu().detach().numpy(), f2.cpu().detach().numpy()) + 1
    
    return ff

def perturb_queries_hardest_multi(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True, rand=False):
    """Perturb the queries with FNA
    
    Arguments:
        q_loader {pytorch dataloader} -- dataloader of the query dataset
        attack {advertorch.attack} -- adversarial attack to perform on the queries
        model {pytorch model} -- pytorch model to evaluate
        device {cuda device} --
        cosine {bool} -- evaluate with cosine similarity
        triplet {bool} -- model trained with triplet or not
        classif {bool} -- model trained with cross entropy
        transforms {bool} -- apply transforms
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()
    probe_features, probe_ids, probe_cams = extract_features(q_loader, model, device, cosine, triplet, classif, transforms)

    guide_features = []

    dict_f = defaultdict(list)
    for f, id in zip(probe_features, probe_ids):
        dict_f[id].append(f)
    
    max_n_id = 0
    for _, values in dict_f.items():
        if len(values) > max_n_id:
            max_n_id = len(values)

    # ## Max distance cluster center
    proto_f = []
    for id, values in dict_f.items():
        proto_f.append(torch.mean(torch.stack(values), dim=0))
    proto_f = torch.stack(proto_f)

    # if not cosine:
    dist_proto = pairwise_distance(probe_features, proto_f)
    keys = list(dict_f.keys()) # keys[i] = id of proto_f[i]
    # max_dist = True

    for nb_q in tqdm(range(len(probe_features))):
        q_d_proto = dist_proto[nb_q]
        id = probe_ids[nb_q]
        guide = []
        for f in dict_f[id]:
            guide.append(f)
        while len(guide) < max_n_id:
            guide.append(probe_features[nb_q])

        if rand: # Choosing a random cluster different from own cluster
            id_f = probe_ids[nb_q] # id current feature
            r = np.random.randint(len(proto_f))
            while keys[r] == id_f: # while proto_f[r] has same id as f
                r = np.random.randint(len(proto_f))
            guide.append(proto_f[r])
        else: # Choosing furthest cluster
            max_d_p = -np.inf
            max_ind_p = 0
            for i,d_p in enumerate(q_d_proto):
                if d_p > max_d_p and probe_ids[nb_q] != keys[i]:
                    max_d_p = d_p
                    max_ind_p = i
            guide.append(proto_f[max_ind_p])
        guide_features.append(torch.stack(guide)) 

    guide_features = torch.stack(guide_features)
    b_s = q_loader.batch_size
    for guides, data in tqdm(zip(torch.split(guide_features, b_s), q_loader), total=len(q_loader)):
        epsilon = attack.eps
        attack.clip_max = data['image'].max()
        attack.clip_min = data['image'].min()
        image_adv = attack.perturb(data['image'].to(device), guides.to(device))
        with torch.no_grad():
            output = extract_feature_img(model, image_adv.to(device), cosine, triplet, classif, transforms)
        features.append(output)
        ids.append(data['id'].cpu())
        cams.append(data['cam'].cpu())

    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)
    # print(f"\ntotal time : {sum(T)}, mean time : {np.mean(T)}" )

    return features.cpu(), ids.numpy(), cams.numpy()

def perturb_queries_hardest_single(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True):
    """Perturb the queries with a single image in FNA
    
    Arguments:
        q_loader {pytorch dataloader} -- dataloader of the query dataset
        attack {advertorch.attack} -- adversarial attack to perform on the queries
        model {pytorch model} -- pytorch model to evaluate
        device {cuda device} --
        cosine {bool} -- evaluate with cosine similarity
        triplet {bool} -- model trained with triplet or not
        classif {bool} -- model trained with cross entropy
        transforms {bool} -- apply transforms
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()
    probe_features, probe_ids, probe_cams = extract_features(q_loader, model, device, cosine, triplet, classif, transforms)

    guide_features = []

    dict_f = defaultdict(list)
    for f, id in zip(probe_features, probe_ids):
        dict_f[id].append(f)
    
    max_n_id = 0
    for _, values in dict_f.items():
        if len(values) > max_n_id:
            max_n_id = len(values)

    ## Max distance
    dist = pairwise_distance(probe_features, probe_features)
    for f_x, d_q, id in tqdm(zip(probe_features, dist, probe_ids), total=len(probe_features)):
        guide = []
        for f in dict_f[id]:
            guide.append(f)
        while len(guide) < max_n_id:
            guide.append(f_x)
        max_d = -np.inf
        max_ind = 0
        for j,d in enumerate(d_q):
            if d > max_d and probe_ids[j] != id:
                max_d = d
                max_ind = j
        guide.append(probe_features[max_ind])
        guide_features.append(torch.stack(guide))

    guide_features = torch.stack(guide_features)
    b_s = q_loader.batch_size
    for guides, data in tqdm(zip(torch.split(guide_features, b_s), q_loader), total=len(q_loader)):
        epsilon = attack.eps
        attack.clip_max = data['image'].max()
        attack.clip_min = data['image'].min()
        image_adv = attack.perturb(data['image'].to(device), guides.to(device))
        with torch.no_grad():
            output = extract_feature_img(model, image_adv.to(device), cosine, triplet, classif, transforms)
        features.append(output)
        ids.append(data['id'].cpu())
        cams.append(data['cam'].cpu())

    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)
    # print(f"\ntotal time : {sum(T)}, mean time : {np.mean(T)}" )

    return features.cpu(), ids.numpy(), cams.numpy()

def perturb_queries_rand(q_loader, attack, model, device, cosine, triplet, classif, transforms, leave=True):
    """Perturb queries with a random noise

Arguments:
        q_loader {pytorch dataloader} -- dataloader of the query dataset
        attack {advertorch.attack} -- adversarial attack to perform on the queries
        model {pytorch model} -- pytorch model to evaluate
        device {cuda device} --
        cosine {bool} -- evaluate with cosine similarity
        triplet {bool} -- model trained with triplet or not
        classif {bool} -- model trained with cross entropy
        transforms {bool} -- apply transforms
    
    Returns:
        features -- Tensor of the features of the queries
        ids -- numpy array of ids
        cams -- numpy array of camera ids
    """
    ids = []
    cams = []
    features = []
    model.eval()

    for _, data in tqdm(enumerate(q_loader), desc="Extracting features", total=len(q_loader), leave=leave):
        eta = torch.rand_like(data['image'])
        epsilon = attack.eps
        image = data['image'] + epsilon*eta
        with torch.no_grad():
            output = extract_feature_img(model, image.to(device), cosine=cosine, triplet=triplet, classif=classif, transforms=transforms)

        features.append(output)
        ids.append(data['id'].cpu())
        cams.append(data['cam'].cpu())

    ids = torch.cat(ids, 0)
    cams = torch.cat(cams, 0)
    features = torch.cat(features, 0)

    return features.cpu(), ids.numpy(), cams.numpy()