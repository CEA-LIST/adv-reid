# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import os
from shutil import copyfile
from pathlib import Path

# You only need to change this line to your dataset download path
download_path = Path('')

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path / 'sorted'
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# query
query_path = download_path / 'query'
query_save_path = save_path / 'query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not (name[-3:]=='jpg'or name[-3:]=='png'):
            continue
        ID  = name.split('_')
        src_path = query_path / name
        dst_path = query_save_path / ID[0] 
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path / name)


#gallery
gallery_path = download_path / 'bounding_box_test'
gallery_save_path = save_path / 'gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not (name[-3:]=='jpg'or name[-3:]=='png'):
            continue
        ID  = name.split('_')
        src_path = gallery_path / name
        dst_path = gallery_save_path / ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path / name)


#train
train_path = download_path / 'bounding_box_train'
train_save_path = save_path / 'train'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not (name[-3:]=='jpg'or name[-3:]=='png'):
            continue
        ID  = name.split('_')
        src_path = train_path / name
        dst_path = train_save_path / ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path / name)