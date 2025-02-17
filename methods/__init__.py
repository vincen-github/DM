from .dm import DM


METHOD_LIST = ["dm"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "dm":
        return DM
