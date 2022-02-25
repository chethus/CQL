import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

import os.path as osp
class DatasetLoader:

    @classmethod
    def get_dataset(cls, dataset_dir, train=True):
        transform = cls.transform if train else cls.test_transform
        return cls.dataset_cls(
            cls.dataset_subdir(dataset_dir),
            train=train,
            download=True,
            transform=transform
        )
    
    @classmethod
    def dataset_subdir(cls, dataset_dir):
        return osp.join(dataset_dir, cls.__name__)


# MNIST / FMNIST

mnist_transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.1307,), (0.3081,))
                        ])


class MNIST(DatasetLoader):
    dataset_cls = torchvision.datasets.MNIST
    transform = mnist_transform
    test_transform = mnist_transform

class FashionMNIST(DatasetLoader):
    dataset_cls = torchvision.datasets.FashionMNIST
    transform = mnist_transform
    test_transform = mnist_transform

# CIFAR-10

cifar_transform_train = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

cifar_transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

class CIFAR10(DatasetLoader):
    dataset_cls = torchvision.datasets.CIFAR10
    transform = cifar_transform_train
    test_transform = cifar_transform_test


