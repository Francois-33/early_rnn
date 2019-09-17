import sys

import argparse
import datetime
import os
import torch
from utils.trainer import Trainer
import ray
from argparse import Namespace
import torch.optim as optim
import config
from config import TUNE_STORE, TUNE_RUNS, TUNE_BATCHSIZE, TUNE_CPU, TUNE_GPU, EPOCHS, TUNE_EPOCH_CHUNKS, TUNE_MAX_CONCURRENT
import pandas as pd

from utils.prepare_components import prepare_model_and_optimizer, prepare_dataset, prepare_loss_criterion

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from utils.optim import ScheduledOptim

def tune(args):
    args.local_dir = os.path.join(TUNE_STORE, args.model)

    try:
        nruns = ray.tune.Analysis(os.path.join(args.local_dir, args.dataset)).dataframe().shape[0]
        resume=False
        todo_runs = TUNE_RUNS - nruns
        print(f"{nruns} found in {os.path.join(args.local_dir, args.dataset)} starting remaining {todo_runs}")
        if todo_runs <= 0:
            print(f"finished all {TUNE_RUNS} runs. Increase TUNE_RUNS in databases.py if necessary. skipping tuning")
            return

    except ValueError as e:
        print(f"could not find any runs in {os.path.join(args.local_dir, args.dataset)}")
        resume=False
        todo_runs = TUNE_RUNS

    ray.init(include_webui=False)

    space, points_to_evaluate = get_hyperparameter_search_space(args)

    args_dict = vars(args)
    config = {**space, **args_dict}
    args = Namespace(**config)

    algo = HyperOptSearch(
        space,
        max_concurrent=TUNE_MAX_CONCURRENT,
        metric="score",
        mode="max",
        points_to_evaluate=points_to_evaluate,
        n_initial_points=TUNE_MAX_CONCURRENT,
    )

    scheduler = AsyncHyperBandScheduler(metric="score", mode="max", max_t=60,
                                        grace_period=2,
                                        reduction_factor=3,
                                        brackets=4)

    ray.tune.run(
        RayTrainer,
        config=config,
        name=args.dataset,
        num_samples=todo_runs,
        local_dir=args.local_dir,
        search_alg=algo,
        scheduler=scheduler,
        verbose=True,
        reuse_actors=True,
        resume=resume,
        checkpoint_at_end=True,
        #global_checkpoint_period=360,
        checkpoint_score_attr="score",
        keep_checkpoints_num=5,
        resources_per_trial=dict(cpu=TUNE_CPU, gpu=TUNE_GPU))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--model', type=str, help='Model variant. supported DualOutputRNN or Conv1d')

    parser.add_argument('-b', '--batchsize', type=int, default=TUNE_BATCHSIZE, help='Model variant. supported DualOutputRNN or Conv1d')
    parser.add_argument('--train-on', type=str, default="train", help='')
    parser.add_argument('--test-on', type=str, default="valid", help='')

    args, _ = parser.parse_known_args()
    return args

def get_hyperparameter_search_space(args):
    """
    simple state function to hold the parameter search space definitions for experiments

    :param experiment: experiment name
    :return: ray databases dictionary
    """
    if args.model == "DualOutputRNN":
        space =  dict(
            batchsize=TUNE_BATCHSIZE,
            workers=2,
            epochs=30,
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([0]),  # [0, 1, 2, 3, 4]),
            hidden_dims=tune.grid_search([2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
            learning_rate=tune.grid_search([1e-2, 1e-3, 1e-4]),
            dropout=0.5,
            num_layers=tune.grid_search([1, 2, 3, 4]),
            dataset=args.dataset
        )
        return space, None

    elif args.model == "Conv1D":
        space = dict(
            hidden_dims=hp.choice('hidden_dims',[25, 50, 75, 100]),
            num_layers=hp.choice('num_layers',[1,2,3,4,5,6,7,8]),
            dropout=hp.uniform("dropout", 0, 1),
            shapelet_width_increment=hp.choice('shapelet_width_increment',[10, 30, 50, 70]),
            alpha=hp.uniform("alpha", 0, 1),
            epsilon=hp.uniform("epsilon", 0, 15),
            weight_decay=hp.loguniform("weight_decay", -1, -12),
            learning_rate=hp.loguniform("learning_rate", -1, -8))

        model_params = pd.read_csv(config.CONV1D_HYPERPARAMETER_CSV, index_col=0).loc[args.dataset]

        initial_point = dict(
            hidden_dims = int(model_params.hidden_dims),
            num_layers=int(model_params.num_layers),
            dropout=model_params.dropout,
            shapelet_width_increment=int(model_params.shapelet_width_increment),
            alpha=model_params.alpha,
            epsilon=model_params.epsilon,
            weight_decay=model_params.weight_decay,
            learning_rate=model_params.learning_rate
        )

        return space, None #[initial_point]
    else:
        raise ValueError("did not recognize model "+args.model)


class RayTrainer(ray.tune.Trainable):
    def _setup(self, config):

        # one iteration is five training epochs, one test epoch
        self.epochs = EPOCHS // TUNE_EPOCH_CHUNKS

        print(config)

        args = Namespace(**config)
        self.traindataloader, self.validdataloader = prepare_dataset(args)

        nclasses = self.traindataloader.dataset.nclasses
        seqlength = self.traindataloader.dataset.sequencelength
        input_dims = self.traindataloader.dataset.ndims

        self.model, self.optimizer = prepare_model_and_optimizer(args, input_dims, seqlength, nclasses)

        self.criterion = prepare_loss_criterion(args)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if "model" in config.keys():
            config.pop('model', None)
        #trainer = Trainer(self.model, self.traindataloader, self.validdataloader, **databases)

        self.trainer = Trainer(self.model,
                               self.traindataloader,
                               self.validdataloader,
                               self.optimizer,
                               self.criterion,
                               store=args.local_dir,
                               test_every_n_epochs=999,
                               visdomlogger=None)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train five epochs and then infer once. to avoid overhead on these small datasets
        for i in range(self.epochs):
            trainstats = self.trainer.train_epoch(epoch=None)

        stats = self.trainer.test_epoch(self.validdataloader)
        stats["score"] = .5*stats["accuracy"] + .5*(1-stats["earliness"])
        stats.pop("inputs")
        stats.pop("confusion_matrix")
        stats.pop("probas")

        #stats["lossdelta"] = trainstats["loss"] - stats["loss"]
        #stats["trainloss"] = trainstats["loss"]

        return stats

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

if __name__=="__main__":

    args = parse_args()

    tune(args)

    #print("Best databases is", analysis.get_best_config(metric="kappa"))
    #analysis.dataframe().to_csv("/tmp/result.csv")

