import torchvision.transforms as T
import torch

all_config = [
    ("Resize", {'size': (256, 256)}),
    ("RandomChoice", {'transforms': [
        ("RandomVerticalFlip", {'p': 0.5}),
        ("RandomHorizontalFlip", {'p': 0.5}),
    ]}),
    ('RandomOrder', {'transforms': [
        ("RandomVerticalFlip", {'p': 0.5}),
        ("RandomHorizontalFlip", {'p': 0.5}),
    ]}),
    ("RandomApply", {
        "transforms": [
            ("ColorJitter", {"brightness": 0.8,
                             "contrast": 0.8, "saturation": 0.8, 'hue': 0.3}),
            ('GaussianBlur', {
             'kernel_size': 128 // 20 * 2 + 1, 'sigma': (0.5, 2.0)})
        ],
        "p": 0.9,
    }),
    ("RandomApply", {
        "transforms": [
            ('GaussianBlur', {
             'kernel_size': 128 // 20 * 2 + 1, 'sigma': (0.5, 2.0)})
        ],
        "p": 0.9,
    }),
    ("ColorJitter", {"brightness": 0.8,
                     "contrast": 0.8, "saturation": 0.8, 'hue': 0.3}),
    ("Grayscale", {}),
    ("Pad", {'padding': 100}),
    ("RandomAffine", {'degrees': 45}),
    ("RandomGrayscale", {"p": 0.5}),
    ("RandomVerticalFlip", {'p': 0.5}),
    ("RandomHorizontalFlip", {'p': 0.5}),
    ("RandomPerspective", {'distortion_scale': 0.5, 'p': 0.9}),
    ("RandomResizedCrop", {'size': 128, "scale": (0.2, 1.0)}),
    ("RandomRotation", {'degrees': 45}),
    ('GaussianBlur', {'kernel_size': 256 // 20 * 2 + 1, 'sigma': (0.5, 2.0)}),
    ("ToTensor", {}),
    ('Normalize', {'mean': [0.485, 0.456, 0.406],
                   'std':[0.229, 0.224, 0.225]}),
    ('RandomErasing', {'p': 0.5}),
    ("ConvertImageDtype", {'dtype': torch.float}),
    ("Lambda", {'lambd': lambda x: x + 256})
]


def get_transform(name, args):
    #
    # GENERIC
    #
    if name == "Lambda":
        t = T.Lambda(**args)
    #
    # TRANSFORM MULTI ON IMGS ONLY
    #
    elif name == "RandomChoice":
        tt = config_transform(args["transforms"])
        t = T.RandomChoice(tt)
    elif name == "RandomOrder":
        tt = config_transform(args["transforms"])
        t = T.RandomOrder(tt)
    #
    # TRANSFORM MULTI ON TENSOR + PIL IMAGE
    #
    elif name == "RandomApply":
        tt = config_transform(args["transforms"])
        t = T.RandomApply(tt, p=args['p'])

    #
    # TRANSFORM ON TENSOR + PIL IMAGE
    #
    elif name == "ColorJitter":
        t = T.ColorJitter(**args)
    elif name == "Grayscale":
        t = T.Grayscale(**args)
    elif name == "Pad":
        t = T.Pad(**args)
    elif name == "RandomAffine":
        t = T.RandomAffine(**args)
    elif name == "RandomGrayscale":
        t = T.RandomGrayscale(**args)
    elif name == "RandomHorizontalFlip":
        t = T.RandomHorizontalFlip(**args)
    elif name == "RandomPerspective":
        t = T.RandomPerspective(**args)
    elif name == 'RandomResizedCrop':
        t = T.RandomResizedCrop(**args)
    elif name == "RandomRotation":
        t = T.RandomRotation(**args)
    elif name == "RandomVerticalFlip":
        t = T.RandomVerticalFlip(**args)
    elif name == "Resize":
        t = T.Resize(**args)
    elif name == "GaussianBlur":
        t = T.GaussianBlur(**args)
    #
    # TRANSFORM ON TENSOR ONLY
    #
    elif name == "Normalize":
        t = T.Normalize(**args)
    elif name == "RandomErasing":
        t = T.RandomErasing(**args)
    elif name == "ConvertImageDtype":
        t = T.ConvertImageDtype(**args)
    elif name == "ToPILImage":
        t = T.ToPILImage(**args)
    elif name == "ToTensor":
        t = T.ToTensor()
    else:
        raise NotImplementedError(transform_name)
    return t


def config_transform(config):
    transformations = []
    for transform_name, transform_args in config:
        t = get_transform(transform_name, transform_args)
        transformations.append(t)
    return transformations


class Augmentation():
    def __init__(self, config):
        self.config = config
        transform = config_transform(config)
        self.transform = T.Compose(transform)

    def __call__(self, x):
        return self.transform(x)


class SimSiamAugmentation():
    def __init__(self, config, downstream: bool):
        self.config = config
        self.downstream = downstream
        self.transform = Augmentation(config)

    def __call__(self, x):
        if self.downstream:
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2
