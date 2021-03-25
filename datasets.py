from pathlib import Path
import json
#
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import torchvision
from PIL import Image


class COCOVisionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 p_base: str,
                 train=True,
                 transform: callable = None,
                 target_transform: callable = None):

        self.p_base = Path(p_base)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        #
        # SET PATHS
        self.p_annotations = self.p_base / "custom_data"
        self.p_imgs_train = self.p_base / "train2017"
        self.p_imgs_valid = self.p_base / "val2017"
        self.p_anns_train = self.p_annotations / 'train2017' / 'annotations.json'
        self.p_anns_valid = self.p_annotations / 'val2017' / 'annotations.json'

        if self.train:
            self.p_imgs = self.p_imgs_train
            self.p_anns = self.p_anns_train
        else:
            self.p_imgs = self.p_imgs_valid
            self.p_anns = self.p_anns_valid
        #
        assert self.p_imgs.exists()
        assert self.p_anns.exists()
        #
        with open(self.p_anns, 'r') as file:
            self.annotations = json.load(file)
        self.annotations = [v for k, v in self.annotations.items()]
        #

    categories = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "street sign", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle",
        "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "mirror", "dining table",
        "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "blender", "book",
        "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush", "hair brush"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # target
        cat = annotation["categories"]

        # load img
        p_img = self.p_imgs / annotation["file_name"]
        img = Image.open(p_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            cat = self.target_transform(cat)

        return img, cat

    @staticmethod
    def custom_transform_labels(categories: list):
        n_obs = len(categories)
        categories = torch.IntTensor(categories)
        cat_torch = torch.zeros(100, dtype=torch.int)
        cat_torch[0: n_obs] = categories
        return cat_torch

    def category_from_index(self, indices):
        return [self.categories[i - 1] for i in indices]


class MultiMNISTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 p_base,
                 train=True,
                 transform: callable = None,
                 target_transform: callable = None):

        self.p_base = Path(p_base)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.p_train = p_base / "train"
        self.p_valid = p_base / "valid"
        self.p_mini = p_base / "mini"

        if train is True:
            self.p_imgs = self.p_train
        elif train is False:
            self.p_imgs = self.p_valid
        elif train == "mini":
            self.p_imgs = self.p_mini
        else:
            raise Exception("split does not exists")

        assert self.p_imgs.exists()

        self.classes = [f.name for f in self.p_imgs.iterdir() if f.is_dir()]
        self.classes = sorted(self.classes)
        self.n_classes = len(self.classes)

        self.class_map = {idx: clasz for idx, clasz in enumerate(self.classes)}

        # create filelist
        self.samples = []
        for class_idx, class_name in self.class_map.items():
            p_clasz = self.p_imgs / class_name
            for p_file in p_clasz.glob("*.png"):
                self.samples.append((p_file, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        p_img, clasz = self.samples[idx]
        img = Image.open(p_img).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            clasz = self.target_transform(clasz)
        return img, clasz


class DSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, p_base, split, transform, target_transform):
        self.p_base = Path(p_base)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.p_file = self.p_base / filename
        assert self.p_file.exists(), f"Missing file {self.p_file}"

        dataset_zip = np.load(self.p_file, allow_pickle=True, encoding='bytes')
        #
        self.imgs = dataset_zip["imgs"]
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

    def __len__(self):
        return imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        if self.transform:
            img = self.transform(img)
        return img, -1


def split_to_train_valid(split):
    if split == "train":
        split = True
    elif split == "valid":
        split = False
    else:
        raise NotImplementedError(
            "Unknown Split for dataset: {}".format(split))
    return split


def get_dataset(dataset, p_data, transform, target_transform=None, split='train', download=False):
    if dataset == 'mscoco2017':
        ds = COCOVisionDataset(
            p_base=p_data,
            train=split_to_train_valid(split),
            target_transform=target_transform,
            transform=transform)
    elif dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10(
            root=p_data,
            train=split_to_train_valid(split),
            target_transform=target_transform,
            transform=transform,
            download=download)
    elif dataset == 'cifar100':
        ds = torchvision.datasets.CIFAR100(
            root=p_data,
            train=split_to_train_valid(split),
            target_transform=target_transform,
            transform=transform,
            download=download)
    elif dataset == 'stl10':
        ds = torchvision.datasets.STL10(
            root=p_data,
            split=split,
            target_transform=target_transform,
            transform=transform,
            download=download)
    elif dataset == 'mnist':
        ds = torchvision.datasets.MNIST(
            root=p_data,
            train=split_to_train_valid(split),
            transform=transform,
            target_transform=target_transform,
            download=download)
    elif dataset == "celeba":
        ds = torchvision.datasets.CelebA(
            root=p_data,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download)
    elif dataset == "dsprites":
        ds = DSpritesDataset(
            p_base=p_data,
            split=split,
            transform=transform,
            target_transform=target_transform)
    else:
        raise NotImplementedError(dataset)
    return ds
