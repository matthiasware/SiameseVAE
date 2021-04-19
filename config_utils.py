from torch.utils.data import DataLoader
from augmentations import SimSiamAugmentation, Augmentation
from datasets import get_dataset
from dotted_dict import DottedDict
from pathlib import Path
import datetime


def get_dataloaders_from_config(config):
    # Augmentations
    trans_train = SimSiamAugmentation(
        config.augmentations_train, downstream=False)
    trans_down_train = SimSiamAugmentation(
        config.augmentations_train, downstream=True)
    trans_down_valid = SimSiamAugmentation(
        config.augmentations_valid, downstream=True)
    #
    # Datasets
    ds_train = get_dataset(
        dataset=config.dataset,
        p_data=config.p_data,
        transform=trans_train,
        target_transform=None,
        split=config.train_split
    )
    ds_down_train = get_dataset(
        dataset=config.dataset,
        p_data=config.p_data,
        transform=trans_down_train,
        target_transform=None,
        split=config.down_train_split
    )
    ds_down_valid = get_dataset(
        dataset=config.dataset,
        p_data=config.p_data,
        transform=trans_down_train,
        target_transform=None,
        split=config.down_valid_split
    )
    # DataLoader
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    dl_down_train = DataLoader(
        ds_down_train,
        batch_size=config.down_batch_size,
        shuffle=True,
        num_workers=config.down_num_workers,
        drop_last=False,
        pin_memory=True
    )

    dl_down_valid = DataLoader(
        ds_down_valid,
        batch_size=config.down_batch_size,
        shuffle=True,
        num_workers=config.down_num_workers,
        drop_last=False,
        pin_memory=True
    )
    return dl_train, dl_down_train, dl_down_valid


def get_config_template():
    config = DottedDict()
    return config


def add_paths_to_confg(config):
    # run directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fs_run = "run_{}_{}_{}".format(config.dataset, config.backbone, timestamp)

    # checkpoint
    config.fs_ckpt = "model_{}_epoch_{:0>6}.ckpt"

    # train dir
    if config.debug:
        config.p_train = Path(config.p_base) / "tmp" / fs_run
    else:
        config.p_train = Path(config.p_base) / fs_run
    config.p_ckpts = config.p_train / "ckpts"
    config.p_logs = config.p_train / "logs"
