import pandas as pd
import numpy as np
import pathlib
from functools import partial


class RecordDataset:
    
    def __init__(self, folder_path='records', infofile='info.csv', predfile='preds.csv', prefix_path=None, memory=False, logprob=False):
        if prefix_path is not None:
            prefix_path = pathlib.Path(prefix_path)
            folder_path = prefix_path / folder_path
            infofile = prefix_path / infofile
            predfile = prefix_path / predfile
        # load predicitions of some model
        pred_df = pd.read_csv(predfile, index_col=0, header=None)
        self.index = list(pred_df.index)
        self.ys = pred_df.loc[self.index].values.astype(float).squeeze()
        if logprob:
            self.ys = np.log(self.ys / (1 - self.ys + 1e-9))
        # load info
        info_df = pd.read_csv(infofile)
        self.cat_cols = list(info_df[info_df.dtype == 'cat'].col)
        self.num_cols = list(info_df[info_df.dtype == 'num'].col)
        self.vocab_sizes = info_df[info_df.dtype == 'cat'].n.values.astype(int)
        # load records
        self.folder_path = pathlib.Path(folder_path)
        self.memory = memory
        if self.memory:
            self.records = [self.load_df(self.folder_path / f'{name}.csv') for name in self.index]
        else:
            self.records = []
            
    def load_df(self, path):
        df = pd.read_csv(path)
        for i, cc in enumerate(self.cat_cols):
            df[cc] = pd.Categorical(df[cc], categories=list(range(self.vocab_sizes[i])))
        return df
    
    def get_df(self, i):
        if self.memory:
            return self.records[i]
        else:
            name = self.index[i]
            # df = self.load_df(self.folder_path / f'{name}.csv')
            df = partial(self.load_df, self.folder_path / f'{name}.csv')
            return df
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        df = self.get_df(i)
        y = self.ys[i]
        return df, y
        

class RecordDataloader:
    
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
    
    def __len__(self):
        n = len(self.dataset) // self.batch_size
        r = len(self.dataset) % self.batch_size
        if r == 0 or self.drop_last:
            return n
        else:
            return n + 1
    
    def __iter__(self):
        if self.shuffle:
            index = self.rng.permutation(len(self.dataset))
        else:
            index = np.arange(len(self.dataset))
        for i in range(len(self)):
            a = i*self.batch_size
            b = (i+1)*self.batch_size
            batch = [self.dataset[j] for j in index[a:b]]
            yield self.collate_batch(batch)

    def get_infinite_iter(self):
        it = iter(self)
        while True:
            try:
                yield next(it)
            except StopIteration:
                it = iter(self)
                yield next(it)
        
    def collate_batch(self, batch):
        dfs, ys = zip(*batch)
        dfs = list(dfs)
        ys = np.array(ys).astype(float)
        return dfs, ys
