from functools import partial
import argparse
from torchvision import models
from datasets import DS_LIST
from methods import METHOD_LIST


def get_cfg():
    """ generates configuration from user input in console """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--method", type=str, choices=METHOD_LIST, default="dm", help="loss type",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="number of samples (d) generated from each image",
    )

    addf = partial(parser.add_argument, type=float)
    addf("--cj_bright", default=0.4, help="color jitter brightness")
    addf("--cj_contrast", default=0.4, help="color jitter contrast")
    addf("--cj_sat", default=0.4, help="color jitter saturation")
    addf("--cj_hue", default=0.1, help="color jitter hue")
    addf("--cj_prob", default=0.8, help="color jitter probability")
    addf("--gs_prob", default=0.2, help="grayscale probability")
    addf("--crop_s0", default=0.2, help="crop size from")
    addf("--crop_s1", default=1.0, help="crop size to")
    addf("--crop_r0", default=0.75, help="crop ratio from")
    addf("--crop_r1", default=(4 / 3), help="crop ratio to")
    addf("--hf_prob", default=0.5, help="horizontal flip probability")
    addf("--blur_prob", default=0.5, help="gaussian blur probability")
    addf("--kernel_size", default=0.1, help="kernel_size")

    parser.add_argument(
        "--no_lr_warmup",
        dest="lr_warmup",
        action="store_false",
        help="do not use learning rate warmup",
    )
    parser.add_argument(
        "--no_add_bn", dest="add_bn", action="store_false", help="do not use BN in head"
    )
    parser.add_argument("--knn", type=int, default=5, help="k in k-nn classifier")
    parser.add_argument("--fname", type=str, help="load model from file")
    parser.add_argument(
        "--lr_step",
        type=str,
        default="step",
        help="learning rate schedule type",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--adam_l2", type=float, default=5e-7, help="weight decay (L2 penalty)"
    )
    parser.add_argument(
        "--head_layers", type=int, default=2, help="number of FC layers in head"
    )
    parser.add_argument(
        "--head_size", type=int, default=1000, help="size of FC layers in head"
    )
    parser.add_argument(
        "--no_norm", dest="norm", action="store_false", help="don't normalize latents",
    )

    parser.add_argument("--epoch", type=int, default=1000, help="total epoch number")
    parser.add_argument(
        "--eval_every_drop",
        type=int,
        default=5,
        help="how often to evaluate after learning rate drop",
    )
    parser.add_argument(
        "--eval_every", type=int, default=20, help="how often to evaluate"
    )
    parser.add_argument("--emb", type=int, default=384, help="embedding size")
    parser.add_argument(
        "--bs", type=int, default=512, help="number of original images in batch N",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=[x for x in dir(models) if "resn" in x],
        default="resnet18",
        help="encoder architecture",
    )
    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="cifar10")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="dataset workers number",
    )
    parser.add_argument(
        "--clf",
        type=str,
        default="sgd",
        choices=["sgd", "knn"],
        help="classifier for test.py",
    )
    parser.add_argument(
        "--eval_head", action="store_true", help="eval head output instead of model",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="range of reference part",
    )
    parser.add_argument(
        "--Kprime",
        type=float,
        default=384,
        help="number of reference part",
    )
    return parser.parse_args()
