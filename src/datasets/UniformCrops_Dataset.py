import torch
import torch.utils.data
import os
import numpy as np

class BavarianCropsDataset(torch.utils.data.Dataset):

    def __init__(self, root, partition="train", samplet=50):

        self.classweights = np.load(os.path.join(root,partition, "classweights.npy"))
        self.y = np.load(os.path.join(root,partition, "y.npy"))
        self.ndims = int(np.load(os.path.join(root,partition, "ndims.npy")))
        self.sequencelengths = np.load(os.path.join(root,partition, "sequencelengths.npy"))
        self.sequencelength = self.sequencelengths.max()
        self.ids = np.load(os.path.join(root,partition, "ids.npy"))
        #self.dataweights = np.load(os.path.join(root,partition, "dataweights.npy"))
        self.X = np.load(os.path.join(root,partition, "X.npy"), allow_pickle=True)

        self.nclasses = len(np.unique(self.y))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        X = self.X[idx]
        y = np.array(self.y[idx].repeat(self.sequencelength))

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y, idx

class UniformDataset(torch.utils.data.Dataset):

    def __init__(self, root, partition="train"):
        X = np.load(os.path.join(root,"X.npy"))
        y = np.load(os.path.join(root,"y.npy"))
        tt = np.load(os.path.join(root, "tt.npy")).astype(bool) # 0:train 1:test

        if partition in ["train","trainvalid","valid"]:
            self.X = X[~tt]
            self.y = y[~tt]
        elif partition in ["eval","test"]:
            self.X = X[tt]
            self.y = y[tt]

        _, self.sequencelength, self.ndims = self.X.shape
        self.nclasses = len(np.unique(y))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        X = self.X[idx]
        y = np.array(self.y[idx].repeat(self.sequencelength))

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y, idx
