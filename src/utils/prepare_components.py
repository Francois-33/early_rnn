import torch
import numpy as np
#from models.wavenet_model import WaveNetModel
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler, WeightedRandomSampler
from datasets.UniformCrops_Dataset import UniformDataset
from utils.loss import loss_cross_entropy, early_loss_cross_entropy, early_loss_linear, loss_early_reward
from models.DualOutputRNN import DualOutputRNN
from models.ConvShapeletModel import ConvShapeletModel
from datasets.UCR_Dataset import UCRDataset
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import pandas as pd
from utils.visdomLogger import VisdomLogger
from utils.optim import ScheduledOptim
import config
import os
from torch.utils.data import DataLoader

from argparse import Namespace

def prepare_dataset(args):

    if args.dataset == "BavarianCrops":
        #traindataset = BavarianCropsDataset(root=databases.BAVARIAN_CROPS_ROOT, partition=args.train_on)
        #testdataset = BavarianCropsDataset(root=databases.BAVARIAN_CROPS_ROOT, partition=args.test_on)

        traindataset = BavarianCropsDataset(root=config.BAVARIAN_CROPS_ROOT, region=config.BAVARIAN_CROPS_REGION,
                                            partition=args.train_on, classmapping=config.BAVARIAN_CROPS_CLASSMAPPING)
        testdataset = BavarianCropsDataset(root=config.BAVARIAN_CROPS_ROOT, region=config.BAVARIAN_CROPS_REGION,
                                            partition=args.test_on, classmapping=config.BAVARIAN_CROPS_CLASSMAPPING)

    elif args.dataset in config.UNIFORM_CORPS_DATASETS:
        traindataset = UniformDataset(os.path.join(config.UNIFORM_CROPS_ROOT,args.dataset), partition=args.train_on)
        testdataset = UniformDataset(os.path.join(config.UNIFORM_CROPS_ROOT, args.dataset), partition=args.test_on)

    elif args.dataset in config.UCR_DATASETS:
        traindataset = UCRDataset(args.dataset, partition=args.train_on, ratio=config.TRAINVALID_SPLIT_RATIO)
        testdataset = UCRDataset(args.dataset, partition=args.test_on, ratio=config.TRAINVALID_SPLIT_RATIO)
    else:
        raise ValueError("Dataset not recognized! is it among: {}?".format(["BavarianCrops"]+config.UNIFORM_CORPS_DATASETS+config.UCR_DATASETS))

    traindataloader = DataLoader(dataset=traindataset,shuffle=True,batch_size=args.batchsize,
                                 num_workers=config.NUM_WORKERS, pin_memory=True)

    testdataloader = DataLoader(dataset=testdataset,shuffle=False,batch_size=args.batchsize,
                                 num_workers=config.NUM_WORKERS, pin_memory=True)

    return traindataloader, testdataloader


def prepare_model_and_optimizer(args, input_dims, sequence_length, num_classes):

    if args.model == "DualOutputRNN":
        hparams = pd.read_csv(config.DUALOUTPUTRNN_HYPERPARAMETER_CSV, index_col=0)

        if args.dataset not in hparams.index:
            raise ValueError(f"No dataset {args.dataset} found in {config.DUALOUTPUTRNN_HYPERPARAMETER_CSV}")

        hparam = hparams.loc[args.dataset]
        model = DualOutputRNN(input_dim=int(input_dims), nclasses=num_classes, hidden_dims=int(hparam.hidden_dims),
                              num_rnn_layers=int(hparam.num_layers), dropout=hparam.dropout, init_late=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=hparam.learning_rate, weight_decay=hparam.weight_decay)
        #optimizer = ScheduledOptim(optimizer, d_model=model.d_model, n_warmup_steps=500)

        return model, optimizer

    elif args.model == "WaveNet":

        model = WaveNetModel(
                 layers=5,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=args.nclasses,
                 classes=args.nclasses,
                 output_length=1,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 input_dims=args.input_dims,
                 bias=False)

    elif args.model == "Conv1D":
        hparams = pd.read_csv(config.CONV1D_HYPERPARAMETER_CSV, index_col=0)

        if args.dataset not in hparams.index:
            raise ValueError(f"No dataset {args.dataset} found in {config.CONV1D_HYPERPARAMETER_CSV}")

        hparam = hparams.loc[args.dataset]

        model = ConvShapeletModel(num_layers=int(hparam.num_layers),
                                  hidden_dims=int(hparam.hidden_dims),
                                  ts_dim=int(input_dims),
                                  n_classes=int(num_classes),
                                  drop_probability=hparam.dropout,
                                  scaleshapeletsize = False,
                                  seqlength=int(sequence_length),
                                  shapelet_width_increment=int(hparam.shapelet_width_increment))

        optimizer = torch.optim.Adam(model.parameters(), lr=hparam.learning_rate, weight_decay=hparam.weight_decay)


        return model, optimizer

    else:
        raise ValueError("Invalid Model, Please insert either 'DualOutputRNN', or 'Conv1D'")


def prepare_loss_criterion(args):

        ## try to optimize for earliness only when classification is correct
        #if args.loss_mode=="early_reward":
        params = pd.read_csv(config.CONV1D_HYPERPARAMETER_CSV, index_col=0)

        if args.dataset not in params.index:
            raise ValueError(f"No dataset {args.dataset} found in {config.CONV1D_HYPERPARAMETER_CSV}")

        param = params.loc[args.dataset]

        return lambda logprobabilities, pts, targets: loss_early_reward(logprobabilities, pts, targets, param.alpha, param.epsilon)


def prepare_visdom(args):
    return VisdomLogger(env="{}_{}_{}".format(args.dataset, args.model,config.LOSS_MODE.replace("_", "-")))
