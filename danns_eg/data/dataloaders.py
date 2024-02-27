#from data.imagenet_ffcv import ImagenetFfcvDataModule, IMAGENET_MEAN
from danns_eg.data.cifar import get_cifar_dataloaders
from danns_eg.data.mnist import get_sparse_mnist_dataloaders, get_sparse_fashionmnist_dataloaders, get_sparse_remove_one_mnist_dataloaders, get_sparse_remove_one_fashionmnist_dataloaders
from torchvision import transforms

class ToCudaTransform:
    def __call__(self, x):
        return x.to('cuda')

def get_dataloaders(p):
    if p.train.dataset == "imagenet": return get_imagenet_dataloaders(p)
    elif "cifar" in p.train.dataset : return get_cifar_dataloaders(p)
    elif "rm_mnist" in p.train.dataset : return get_sparse_remove_one_mnist_dataloaders(p, rm_digits=[0, 3, 8, 6])
    elif "rm_fashionmnist" in p.train.dataset : return get_sparse_remove_one_fashionmnist_dataloaders(p, rm_items=[5, 9, 7, 0])
    elif "mnist" in p.train.dataset : return get_sparse_mnist_dataloaders(p) #,  transforms=transforms.Compose([
                                #     transforms.ToPILImage(),
                                #     transforms.ToTensor(),
                                #     transforms.Normalize((0.3,), (0.6,)),
                                #     ToCudaTransform()
                                # ]))
    elif "fashionmnist" in p.train.dataset: return get_sparse_fashionmnist_dataloaders(p)
    else:print(f"ERROR: {p.train.dataset} not recognised as a vaild dataset")

def get_imagenet_dataloaders(p):
    "p is a params object"
    if p.exp.use_testset:
        print(" using testset!") 
        imgs_per_val_class = 0
    else: imgs_per_val_class = 50
    
    datamodule = ImagenetFfcvDataModule(num_workers=p.exp.num_workers, 
                                        batch_size=p.train.batch_size,
                                        num_imgs_per_val_class=imgs_per_val_class)
    datamodule.prepare_data()
    dataloaders = {
        "train":datamodule.train_dataloader(),
        "train_eval":datamodule.train_dataloader(transforms="val"),
    }
    if p.exp.use_testset: dataloaders['val'] = datamodule.test_dataloader()
    else:
        print("not here...")
        dataloaders['val'] = datamodule.val_dataloader()

    return dataloaders 

def get_arna_cifar_dataloaders():
    from typing import List

    # Import torch and torchvision
    import torch as ch
    import torchvision

    # Import ffcv packages
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze

    ffcv_datadir = '/network/projects/_groups/linclab_users/ffcv/ffcv_datasets'
    dataset = 'cifar10'
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    NUM_CLASSES = 100 if 'cifar100' in dataset else 10

    BATCH_SIZE = 512
    # ==================================================================================================================================

    # Define the dataloaders
    # ==================================================================================================================================
    loaders = {}
    for name in ['train', 'test']:
        # initialize the labels pipeline
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        # initialize the Image pipeline
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        # Add the conversion to cuda and torch Tensor transformations (same for train and validation/test) 
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'{ffcv_datadir}/{dataset}/{name}.beton',
                                batch_size=BATCH_SIZE,
                                num_workers=4, 			# Change the number of workers here if you are using 1 cpu	
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                        'label': label_pipeline})

    return loaders 
    # ==============================================================