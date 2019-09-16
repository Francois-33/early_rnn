import os
from argparse import Namespace

# -d BavarianCrops -m DualOutputRNN classmapping /data/BavarianCrops/classmapping.csv.holl
BavarianCrops_dataset = Namespace(
    dataset = "BavarianCrops",
    classmapping = "/data/BavarianCrops/classmapping.csv.holl",
    trainregions = ["holl"],
    testregions = ["holl"],
    mode="traintest",
    test_on = "test",
    train_on = "train",
    samplet = 70
)

DualOutputRNN_model = Namespace(
    model = "GAFv2",
)

WaveNet_model = Namespace(
    model = "WaveNet",
)


def experiments(args):

    """Experiment Modalities"""
    if args.experiment == "rnn":
        args = merge([args, GAF_dataset, hyperparameters_rnn])
        args.features="optical"

    else:
        raise ValueError("Wrong experiment name!")

    return args


def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)