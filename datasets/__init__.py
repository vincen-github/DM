from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .stl10 import STL10


DS_LIST = ["cifar10", "cifar100", "stl10"]


def get_ds(name):
    assert name in DS_LIST
    if name == "cifar10":
        return CIFAR10
    elif name == "cifar100":
        return CIFAR100
    elif name == "stl10":
        return STL10
