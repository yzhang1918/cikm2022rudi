import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial


pool = None



def pipe_pre_select_parallel(actions, dfs, cpucore):
    if cpucore == 1:
        results = [pipe_pre_select(actions, df) for df in dfs]
        return results
    else:
        global pool
        if pool is None:
            pool = mp.Pool(cpucore)
        # results = pool.map(partial(pipe_pre_select, actions), dfs)
        results = list(pool.imap(partial(pipe_pre_select, actions), dfs, chunksize=16))
        # with mp.Pool(cpucore) as pool:
        #     results = pool.map(partial(pipe_pre_select, actions), dfs)
        return results


def pipe(actions, x, df):
    dim = None
    if x.dtype.name == 'category':
        for i, op in enumerate(actions):
            if isinstance(op, GroupBy) and not isinstance(actions[i+1], Count):
                x = pd.get_dummies(x).astype(float)
                dim = x.shape[1]
                index = x.columns
                break
        for op in actions:
            if isinstance(op, Count):
                break
        else:
            x = pd.get_dummies(x).astype(float)
            dim = x.shape[1]
            index = x.columns
    for op in actions:
        x, df = op(x, df)
    if isinstance(x, float) and not np.isfinite(x) and dim is not None:
        x = pd.Series(np.full(dim, np.nan), index=index)
    return x


def pipe_pre_select(actions, df):
    if not isinstance(df, pd.DataFrame):
        df = df()
    select_op = actions[0]
    x = select_op(df)
    return pipe(actions[1:], x, df)


def pipe_post_select(actions, df):
    select_op = actions[-1]
    x = select_op(df)
    return pipe(actions[:-1], x, df)


class SelectOp:
    def __init__(self, col, dtype):
        # self.terminal = False
        # self.destructive = False
        self.col = col
        self.dtype = dtype

    def __call__(self, df: pd.DataFrame):
        x = df[self.col].copy()
        return x

    def valid_path(self, path):
        pass

    def __repr__(self):
        return f'Select({self.col})'

    __str__ = __repr__


class UnaryOp:
    def __init__(self):
        self.terminal = False
        self.destructive = False

    def __call__(self, x: pd.Series, df: pd.DataFrame = None):
        pass

    def valid_path(self, path):
        return True

    def __repr__(self):
        return self.__class__.__name__ + '()'

    __str__ = __repr__


class AggerateOp(UnaryOp):
    agg_name = None

    def __init__(self):
        self.terminal = True
        self.destructive = True

    def __call__(self, x, df: pd.DataFrame = None):
        if isinstance(x, float) or len(x) == 0:
            return np.nan, None
        return x.agg(self.agg_name), None

    def valid_path(self, path):
        if len(path) and isinstance(path[-1], GroupBy):
            return True
        for op in path:
            if isinstance(op, SortBy):
                return False
        else:
            return True


class BinaryOp:
    def __init__(self, by_col):
        self.by_col = by_col
        self.terminal = False
        self.destructive = False

    def __call__(self, x: pd.Series, df: pd.DataFrame = None):
        pass

    def valid_path(self, path):
        return True

    def __repr__(self):
        return self.__class__.__name__ + f'({self.by_col})'

    __str__ = __repr__


class TernaryOp:
    def __init__(self, by_col, cond):
        self.by_col = by_col
        self.cond = cond
        self.terminal = False
        self.destructive = False

    def __repr__(self):
        return self.__class__.__name__ + f'({self.by_col}, {self.cond})'

    __str__ = __repr__

    def __call__(self, x: pd.Series, df: pd.DataFrame = None):
        pass

    def valid_path(self, path):
        return True


class Min(AggerateOp):
    agg_name = 'min'


class Max(AggerateOp):
    agg_name = 'max'


class Sum(AggerateOp):
    agg_name = 'sum'


class Mean(AggerateOp):
    agg_name = 'mean'


class Std(AggerateOp):
    agg_name = 'std'


class Ptp(AggerateOp):
    q = None

    def __call__(self, x, *args):
        if isinstance(x, float) or len(x) == 0:
            return np.nan, None
        if isinstance(x, pd.Series):
            return x.agg('ptp'), None
        else:  # GroupyBy
            if np.all(x.count() == 0):
                return x.agg(np.max), None  # all nan, expedient
            else:
                return x.agg(np.ptp), None


class Count(AggerateOp):
    agg_name = 'count'


class Percentile(AggerateOp):
    q = None

    def __call__(self, x, *args):
        if isinstance(x, float) or len(x) == 0:
            return np.nan, None
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            # if x.dtype.name == 'category':
            #     x = pd.get_dummies(x)
            return x.agg(np.percentile, 0, self.q), None
        else:  # GroupBy
            return x.agg(np.percentile, self.q), None

    def valid_path(self, path):
        # GroupBy -X> Percentile
        if len(path) and isinstance(path[-1], GroupBy):
            return False
        for op in path:
            if isinstance(op, SortBy):
                return False
        else:
            return True


class Percentile5(Percentile):
    q = 5


class Percentile10(Percentile):
    q = 10


class Percentile25(Percentile):
    q = 25


class Percentile50(Percentile):
    q = 50


class Percentile75(Percentile):
    q = 75


class Percentile90(Percentile):
    q = 90


class Percentile95(Percentile):
    q = 95


class Abs(UnaryOp):
    def __call__(self, x: pd.Series, df: pd.DataFrame):
        return x.abs(), df

    def valid_path(self, path):
        for op in path:
            if isinstance(op, Abs) or isinstance(op, GroupBy):
                return False
        else:
            return True


class Top1(UnaryOp):
    def __init__(self):
        super().__init__()
        self.terminal = True
        self.destructive = True

    def __call__(self, x: pd.Series, df: pd.DataFrame):
        if len(x) == 0:
            return np.nan, None
        return x.iloc[0], None

    def valid_path(self, path):
        if len(path) and isinstance(path[-1], GroupBy):
            return False
        for op in path:
            if isinstance(op, SortBy):
                return True
        else:
            return False


class Top5(UnaryOp):
    k = 5

    def __call__(self, x: pd.Series, df: pd.DataFrame):
        return x.iloc[:self.k], df.iloc[:self.k]

    def valid_path(self, path):
        if len(path) and isinstance(path[-1], GroupBy):
            return False
        for op in path:
            if isinstance(op, SortBy):
                return True
        else:
            return False


class GroupBy(BinaryOp):

    def __init__(self, by_col):
        super().__init__(by_col)
        self.destructive = True

    def __call__(self, x: pd.Series, df: pd.DataFrame):
        return x.groupby(df[self.by_col]), None

    def valid_path(self, path):
        for op in path:
            if isinstance(op, GroupBy):
                return False
            if (isinstance(op, FilterBy) or isinstance(op, RetainBy)) and op.by_col == self.by_col:
                return False
        else:
            return True


class FilterBy(TernaryOp):
    def __call__(self, x: pd.Series, df: pd.DataFrame):
        idx = (df[self.by_col] != self.cond)
        return x[idx], df[idx]

    def valid_path(self, path):
        for op in path:
            if op.destructive:
                return False
            elif (isinstance(op, FilterBy) or isinstance(op, RetainBy)) and op.by_col == self.by_col and op.cond == op.cond:
                return False
        return True


class RetainBy(TernaryOp):
    def __call__(self, x: pd.Series, df: pd.DataFrame):
        idx = (df[self.by_col] == self.cond)
        return x[idx], df[idx]

    def valid_path(self, path):
        for op in path:
            if op.destructive:
                return False
            elif isinstance(op, RetainBy) and op.by_col == self.by_col:
                return False
            elif isinstance(op, FilterBy) and op.by_col == self.by_col and op.cond == op.cond:
                return False
        return True


class SortBy(TernaryOp):
    def __init__(self, by_col, ascending=True):
        super().__init__(by_col, ascending)
        self.ascending = ascending

    def __call__(self, x: pd.Series, df: pd.DataFrame):
        if self.by_col == '__self__':
            idx = np.argsort(x)
        else:
            idx = np.argsort(df[self.by_col])
        if not self.ascending:
            idx = idx[::-1]
        if df is None:
            return x.iloc[idx], None
        else:
            return x.iloc[idx], df.iloc[idx]

    def valid_path(self, path):
        if len(path) and isinstance(path[-1], GroupBy):  # GroupBy -X> SortBy
            return False
        elif self.by_col == '__self__':
            for op in path:
                if isinstance(op, SortBy) and op.by_col == '__self__':
                    return False
            return True
        for op in path:
            if op.destructive:
                return False
            if isinstance(op, SortBy) and op.by_col == self.by_col:
                return False
        else:
            return True


terminal_ops = [Min, Max, Sum, Mean, Std, Ptp, Count, Percentile5, Percentile10,
                Percentile25, Percentile50, Percentile75, Percentile90, Percentile95]
tsfm_op = [Abs]
top_ops = [Top1, Top5]
by_ops = [GroupBy, FilterBy, RetainBy, SortBy]
