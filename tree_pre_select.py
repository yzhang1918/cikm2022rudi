from ops import *


class Node:
    def __init__(self, parent=None, action=None, candidates=None, pre_path=None, popped_paths=None, c=0.7071, max_depth=3):
        self.Q = 0.
        self.N = 0
        self.parent = parent
        self.action = action
        if self.parent is None:
            self.path = [] if pre_path is None else pre_path
            self.c = c
            self.max_depth = max_depth
            self.candidates = candidates
            self.popped_paths = popped_paths
        else:
            self.path = parent.path + [action]
            self.c = parent.c
            self.max_depth = parent.max_depth
            self.candidates = parent.candidates
            self.parent.children.append(self)
            self.popped_paths = parent.popped_paths
            
        self.untried_actions = get_valid_ops_pre_select(self.path, self.candidates, self.max_depth, self.popped_paths)
        np.random.shuffle(self.untried_actions)
        self.children = []
        self.terminal = len(self.untried_actions) == 0
        
        
    def __str__(self):
        return f'Node{self.path}(mu={self.Q/(self.N+1e-5):.4f} N={self.N})'
    
    __repr__ = __str__
    
    def rollout_actions(self):
        path = self.path.copy()
        while True:
            next_actions = get_valid_ops_pre_select(path, self.candidates, self.max_depth, self.popped_paths)
            if len(next_actions) == 0:
                break
            else:
                path.append(np.random.choice(next_actions))
        return path

    
    def score(self, n_parent_visit, c):
        if self.N == 0:  # should only happen after popping nodes
            return self.Q
            # raise ValueError()
            # return np.inf
        else:
            return self.Q / self.N + c * (2 * np.log(n_parent_visit) / self.N) ** .5
    
    def best_child(self, c=None):
        if c is None:
            c = self.c
        scores = [node.score(self.N, c) for node in self.children]
        i = np.argmax(scores)
        return self.children[i]

        
def get_valid_ops(path, candidates, max_length):
    valid_ops = []
    if len(path) == 1 and path[0].terminal:  # root -> Agg
        return []
    elif len(path) > 1:
        if path[-1].terminal and not isinstance(path[-2], GroupBy):  # ... -> (not groupby) -> Agg
            return []
    for op in candidates:
        if op.valid_path(path):
            valid_ops.append(op)
    if len(path) == max_length - 1:  # the last layer: only terminal ops
        valid_ops = [op for op in valid_ops if op.terminal]
    elif len(path) == max_length - 2:  # the second last layer: no groupby ops
        valid_ops = [op for op in valid_ops if not isinstance(op, GroupBy)]
    return valid_ops


def get_valid_ops_pre_select(path, candidates, max_length, popped_paths=None):
    if len(path) == 0:
        assert max_length >= 2
        valid_ops = [op for op in candidates if isinstance(op, SelectOp)]
    else:
        select_op = path[0]
        target_col = select_op.col
        candidates = [op for op in candidates if not isinstance(op, SelectOp)]
        if select_op.dtype == 'num':
            candidates = [op for op in candidates 
                          if isinstance(op, UnaryOp) or op.by_col != target_col]
        elif select_op.dtype == 'cat':
            candidates = [op for op in candidates if isinstance(op, UnaryOp) 
                          or (op.by_col != target_col and op.by_col != '__self__')]
            candidates = [op for op in candidates if not isinstance(op, Abs)]
        else:
            raise NotImplementedError
        valid_ops = get_valid_ops(path[1:], candidates, max_length-1)
    # remove popped ops
    if popped_paths is not None:
        valid_ops = [op for op in valid_ops if repr_path(*path, op) not in popped_paths]
    return valid_ops
    

def backup(node, reward):
    while node is not None:
        node.Q += reward
        node.N += 1
        node = node.parent
        
def tree_policy(node):
    while not node.terminal:
        if len(node.untried_actions):
            # expand
            action = node.untried_actions.pop()
            node = Node(node, action)
            return node
        else:
            node = node.best_child()
    return node


def default_policy(node, dfs, ys, cpucore):
    path = node.rollout_actions()
    if path[0].dtype == 'cat':
        # stats = [pipe_pre_select(path, df) for df in dfs]  # vanilla
        stats = pipe_pre_select_parallel(path, dfs, cpucore)
        mask = [isinstance(x, pd.Series) and np.isfinite(x).all() for x in stats]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return -1
        else:
            A = pd.concat([stats[i] for i in idx], 1).T.values
            A = np.concatenate([A, np.ones_like(A[:, :1])], 1)
            b = ys[idx]
            beta, *_ = np.linalg.lstsq(A.astype(float), b, rcond=None)
            y = A @ beta
            corr = np.corrcoef(y, b)[0, 1]
            reward = abs(corr)
        if not np.isfinite(reward):
            reward = -1
        return reward
    else:
        # stats = np.array([pipe_pre_select(path, df) for df in dfs])
        stats = np.array(pipe_pre_select_parallel(path, dfs, cpucore))
        idx = np.isfinite(stats)
        if len(idx) == 0:
            return -1
        else:
            corr = np.corrcoef(stats[idx], ys[idx])[0, 1]
            reward = abs(corr)
        if not np.isfinite(reward):
            reward = -1
        return reward


def repr_path(*args):
    s = '.'.join([str(op) for op in args])
    return s
