import cv2
import os
import numpy as np
from tqdm import tqdm
import torchvision
import torch
from torch.utils.data import random_split

import warnings
from ffcv.writer import DatasetWriter
from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, NormalizeImage
from ffcv.transforms.common import Squeeze

from pathlib import Path
from multiprocessing import cpu_count

"""
How to make a validationsplit? plbolts?
the imagenet code's writing timeline
"""

# is this for cifar 10 or 100? 


def get_slurm_tmpdir() -> Path:
    """Returns the SLURM temporary directory. Also works when using `mila code`."""
    if "SLURM_TMPDIR" in os.environ:
        return Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_JOB_ID" in os.environ:
        # Running with `mila code`, so we don't have all the SLURM environment variables.
        # However, we're lucky, because we can reliably predict the SLURM_TMPDIR given the job id.
        job_id = os.environ["SLURM_JOB_ID"]
        return Path(f"/Tmp/slurm.{job_id}.0")
    raise RuntimeError(
        "None of SLURM_TMPDIR or SLURM_JOB_ID are set! "
        "Cannot locate the SLURM temporary directory."
    )

def get_cpus_on_node() -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return cpu_count()

def get_datadir(p):
    """
    Function returns location to write/ load the data
    At the moment we are using slurm tmodir,
    """
    slurm_tmpdir = get_slurm_tmpdir()
    assert slurm_tmpdir
    data_dir = slurm_tmpdir / p.train.dataset
    if p.exp.data_dir != "" and p.exp.data_dir != data_dir:
        warnings.warn(
            RuntimeWarning(
                f"Ignoring passed data_dir ({p.exp.data_dir}), using {data_dir} instead."
                )
            )
    return data_dir

def write_ffcv_dataset(p):
    data_dir = get_datadir(p)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if p.train.dataset=='cifar10':
        source_dataset_folder = '/network/datasets/cifar10.var/cifar10_torchvision/'
        trainset = torchvision.datasets.CIFAR10(root=source_dataset_folder, train=True, download=False, transform=None)
        testset = torchvision.datasets.CIFAR10(root=source_dataset_folder, train=False, download=False, transform=None)
    
    elif p.train.dataset=='cifar100':
        source_dataset_folder = '/network/datasets/cifar100.var/cifar100_torchvision/'
        trainset = torchvision.datasets.CIFAR100(root=source_dataset_folder, train=True, download=False, transform=None)
        testset = torchvision.datasets.CIFAR100(root=source_dataset_folder, train=False, download=False, transform=None)
    
    if p.train.use_testset:
        datasets = {'train': trainset, 'test':testset}
    else:
        # hardcoding a split of 10k, as 50k training 10k test
        train_split, val_spit = random_split(trainset, (40000,10000))
        datasets = {'train': train_split, 'test':val_spit}

    train_beton_fpath = os.path.join(data_dir,'train.beton')
    test_beton_fpath = os.path.join(data_dir,'test.beton')
    for name,ds in datasets.items():
        path = train_beton_fpath if name=='train' else test_beton_fpath
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
            })
        writer.from_indexed_dataset(ds)

    return train_beton_fpath, test_beton_fpath


# from ffcv example https://github.com/libffcv/ffcv/blob/main/examples/cifar/train_cifar.py
def get_cifar_dataloaders(p):
    # assumes beton files on slurm? 
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    if p.data.subtract_mean == False:
        CIFAR_MEAN = [0.0, 0.0, 0.0]
        print("Info: not subtracting the mean!")

    train_dataset, test_dataset= write_ffcv_dataset(p)
    paths = {
        'train': train_dataset,
        'train_eval': train_dataset,
        'test': test_dataset
    }
    loaders = {}
    for name in ['train', 'test', 'train_eval']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        # if name == 'train':
        #     image_pipeline.extend([
        #         RandomHorizontalFlip(),
        #         RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
        #         Cutout(4, tuple(map(int, CIFAR_MEAN))),
        #     ])
        if name == 'train': # in arna's example
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        # if not p.exp.use_autocast: Convert(torch.float32),
        # getting error RuntimeError: result type Float can't be cast to the desired output type Byte
    
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=p.train.batch_size, 
                               num_workers=p.exp.num_workers,
                               order=ordering, 
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})


    # for key, loader in loaders.items():
    #     print(key)
    #     print(len(loader), loader.batch_size, type(loader))
    #     if key == "val":
    #         print(loader.dataset)
    #     x,y = next(iter(loader))
    #     print(x.shape)
    #     print(y.shape)

    return loaders