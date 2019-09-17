from argparse import Namespace
"""
A python configuration file
"""

TRAINVALID_SPLIT_RATIO=0.75

# Dataloader workers
NUM_WORKERS=1

# Hyperparamter files
CONV1D_HYPERPARAMETER_CSV = "/home/marc/projects/early_rnn/databases/hyperparameter/Conv1D.csv"
DUALOUTPUTRNN_HYPERPARAMETER_CSV = "/home/marc/projects/early_rnn/databases/hyperparameter/DualOutputRNN.csv"

# BavarianCrops Dataset
BAVARIAN_CROPS_REGION = "HOLL_2018_MT_pilot"
BAVARIAN_CROPS_CLASSMAPPING = "/data/BavarianCrops/classmapping.csv.holl"
BAVARIAN_CROPS_ROOT = "/data/BavarianCrops"

# Uniform Crops Dataset
# class uniformly sampled versions of the BavarianCrops Dataset
UNIFORM_CORPS_DATASETS = ["BavarianCrops_uniform_25",
                          "BavarianCrops_uniform_50",
                          "BavarianCrops_uniform_75",
                          "BavarianCrops_uniform_100",
                          "BavarianCrops_uniform_250",
                          "BavarianCrops_uniform_500",
                          "BavarianCrops_uniform_750",
                          "BavarianCrops_uniform_1000",
                          "BavarianCrops_uniform_2500",
                          "BavarianCrops_uniform_5000"]
UNIFORM_CROPS_ROOT = "/data/BavarianCropsHoll8"

# UCR Datasets
UCR_DATASETS = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration',
                'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction',
                'DistalPhalanxOutlineCorrect', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'Earthquakes',
                'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
                'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
                'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
                'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup',
                'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2',
                'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect',
                'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType',
                'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
                'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1',
                'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX',
                'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'UWaveGestureLibraryAll', 'Wafer', 'Wine',
                'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

TUNE_STORE = "/data/early_rnn/tune"
TUNE_RUNS = 300
TUNE_BATCHSIZE=32
TUNE_EPOCH_CHUNKS = 20
TUNE_CPU=2
TUNE_GPU=.2
TRAIN_STORE = "/data/early_rnn/train"

TRAIN_BATCHSIZE=128

EPOCHS = 100

LOSS_MODE = "early_reward"

RESULT_ACCURACY_CSV = "/home/marc/projects/early_rnn/databases/sota_accuracy.csv"
RESULT_EARLINESS_CSV = "/home/marc/projects/early_rnn/databases/sota_earliness.csv"
