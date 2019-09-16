import argparse
from utils.trainer import Trainer
from utils.prepare_components import prepare_model_and_optimizer, prepare_dataset, prepare_loss_criterion, prepare_visdom
import os
import torch
import config

def train(args):

    traindataloader, testdataloader = prepare_dataset(args)

    model, optimizer = prepare_model_and_optimizer(args,
                                                   traindataloader.dataset.ndims,
                                                   traindataloader.dataset.sequencelength,
                                                   traindataloader.dataset.nclasses)

    criterion = prepare_loss_criterion(args)

    trainer = Trainer(model.cuda() if torch.cuda.is_available() else model,
                      traindataloader,
                      testdataloader,
                      optimizer,
                      criterion,
                      visdomlogger=prepare_visdom(args) if not args.no_visdom else None,
                      store=os.path.join(config.TRAIN_STORE, args.model, args.dataset),
                      test_every_n_epochs=args.test_every_n_epochs)

    trainer.fit(config.EPOCHS)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--model', type=str, help='Model variant. supported DualOutputRNN or Conv1d')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch Size')

    parser.add_argument('--train-on', type=str, default="trainvalid", help='training partition')
    parser.add_argument('--test-on', type=str, default="test", help='testing partition')
    parser.add_argument('--no-visdom', action='store_true',help="dont send visdom logs")
    parser.add_argument('--test_every_n_epochs', type=int, default=1, help='skip some test epochs for faster overall training')

    args, _ = parser.parse_known_args()

    return args


if __name__=="__main__":
    args = parse_args()
    train(args)
