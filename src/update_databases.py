import config
import pandas as pd
import ray.tune as tune
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--model', type=str, help='Model')
    args, _ = parser.parse_known_args()

    return args

def update_hyperparameter_csv(model, dataset):

    ray_run_directory = os.path.join(config.TUNE_STORE, model, dataset)

    best = tune.Analysis(ray_run_directory).dataframe().sort_values(by="score", ascending=False)
    best.columns = [col.replace("config/", "") for col in best.columns]
    best = best.iloc[0]

    if model == "Conv1D":
        hparam = ["hidden_dims", "learning_rate", "num_layers", "shapelet_width_increment", "dropout", "weight_decay","alpha", "epsilon"]
    else:
        raise NotImplementedError()

    # read csv
    print(f"opening {config.CONV1D_HYPERPARAMETER_CSV}")
    hyperparameter_database = pd.read_csv(config.CONV1D_HYPERPARAMETER_CSV, index_col=0)

    # update row
    hyperparameter_database.loc[dataset] = best[hparam]
    print(f"updating {dataset} with {best[hparam].to_dict()}")

    # store
    print(f"saving {config.CONV1D_HYPERPARAMETER_CSV}")
    hyperparameter_database.to_csv(config.CONV1D_HYPERPARAMETER_CSV, float_format="%.8f")

def update_result_csv(model, dataset):

    run_directory = os.path.join(config.TRAIN_STORE, model, dataset)

    log = pd.read_csv(os.path.join(run_directory,"data.csv"), index_col=0)

    test = log.loc[log["mode"]=="test"]

    score = .5 * test["accuracy"] + .5 * (1 - test["earliness"])

    best_idx = score.idxmax()

    best_epoch = test.loc[best_idx]

    # read csv
    print(f"opening {config.RESULT_ACCURACY_CSV}")
    accuracy_database = pd.read_csv(config.RESULT_ACCURACY_CSV, index_col=0)
    print(f"updating {dataset} with {best_epoch.accuracy}")
    accuracy_database.loc[dataset,"ELECTS"] = best_epoch.accuracy*100
    print(f"saving {config.RESULT_ACCURACY_CSV}")
    accuracy_database.to_csv(config.RESULT_ACCURACY_CSV, float_format="%.8f")

    print(f"opening {config.RESULT_EARLINESS_CSV}")
    accuracy_database = pd.read_csv(config.RESULT_EARLINESS_CSV, index_col=0)
    print(f"updating {dataset} with {best_epoch.earliness}")
    accuracy_database.loc[dataset,"ELECTS"] = best_epoch.earliness*100
    print(f"saving {config.RESULT_EARLINESS_CSV}")
    accuracy_database.to_csv(config.RESULT_EARLINESS_CSV, float_format="%.8f")

if __name__=="__main__":
    args = parse_args()
    update_hyperparameter_csv(args.model, args.dataset)
    update_result_csv(args.model, args.dataset)