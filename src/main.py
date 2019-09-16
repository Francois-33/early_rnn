from train import train
from tune import tune
from update_databases import update_hyperparameter_csv, update_result_csv
import argparse
from config import TUNE_BATCHSIZE, TRAIN_BATCHSIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--model', type=str, help='Model variant. supported DualOutputRNN or Conv1d')

    args, _ = parser.parse_known_args()

    # tune hyperparameter
    args.train_on = "train"
    args.test_on = "valid"
    args.batchsize = TUNE_BATCHSIZE
    tune(args)

    # update hyperparameter csv
    update_hyperparameter_csv(args.model, args.dataset)

    # train and evaluate
    args.train_on = "trainvalid"
    args.test_on = "test"
    args.no_visdom = False
    args.test_every_n_epochs = 1
    args.batchsize = TRAIN_BATCHSIZE

    train(args)

    # update sota csv files
    update_result_csv(args.model, args.dataset)

if __name__=="__main__":
    main()