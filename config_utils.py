from torch.utils.data import DataLoader
from augmentations import SimSiamAugmentation, Augmentation
from datasets import get_dataset


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
        split='train'
    )
    ds_down_train = get_dataset(
        dataset=config.dataset,
        p_data=config.p_data,
        transform=trans_down_train,
        target_transform=None,
        split='train'
    )
    ds_down_valid = get_dataset(
        dataset=config.dataset,
        p_data=config.p_data,
        transform=trans_down_train,
        target_transform=None,
        split='valid'
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
