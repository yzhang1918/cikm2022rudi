import tqdm
from ops import *
from tree_pre_select import *


def init_ops(ds):
    all_ops = []
    for col in ds.cat_cols:
        all_ops.append(SelectOp(col, 'cat'))
    for col in ds.num_cols:
        all_ops.append(SelectOp(col, 'num'))
    for OP in terminal_ops + tsfm_op + top_ops:
        all_ops.append(OP())
    for c, size in zip(ds.cat_cols, ds.vocab_sizes):
        all_ops.append(GroupBy(c))
        for cond in range(size):
            all_ops.append(FilterBy(c, cond))
            all_ops.append(RetainBy(c, cond))
    for cond in [True, False]:
        all_ops.append(SortBy('__self__', cond))
        for c in ds.num_cols:
            all_ops.append(SortBy(c, cond))
    return all_ops

    
def get_best_action(root, it, iterations, cpucore, depth):
    for i in tqdm.trange(iterations, desc=f'depth {depth}', leave=False):
        dfs, ys = next(it)
        node = tree_policy(root)
        with np.errstate(divide='ignore', invalid='ignore'):
            reward = default_policy(node, dfs, ys, cpucore)
        backup(node, reward)
    return root.best_child(c=0)


def get_best_path(root, it, iterations, cpucore):
    node = root
    depth = 1
    while not node.terminal:
        node = get_best_action(node, it, iterations, cpucore, depth)
        depth += 1
    return node


def get_topk_stats_sequential(ops, max_depth, dl, iterations=5, k=3, cpucore=1):
    it = dl.get_infinite_iter()
    results = []
    stats = []
    header = []
    popped_paths = set()
    dl.dataset._backup_ys = dl.dataset.ys
    for i in tqdm.trange(k, desc='Stats'):
        root = Node(candidates=ops, max_depth=max_depth, popped_paths=popped_paths)
        node = get_best_path(root, it, iterations, cpucore)
        results.append(node)
        popped_paths.add(repr_path(*node.path))
        # compute stats and residuals
        # current_stats = [pipe_pre_select(node.path, df) for df, _ in dl.dataset]
        current_stats = pipe_pre_select_parallel(node.path, (x[0] for x in dl.dataset), cpucore)
        current_stats = np.asarray([s.values if isinstance(s, pd.Series) else s for s in current_stats])
        current_stats = np.nan_to_num(current_stats).reshape(len(dl.dataset), -1)
        dim = current_stats.shape[1]
        if dim == 1:  # header of generated stats csv files.
            header.append(f'R{i}')
        else:
            header.extend([f'R{i}D{j}' for j in range(dim)])
        stats.append(current_stats)
        A = np.concatenate(stats, 1)
        beta, *_ = np.linalg.lstsq(A.astype(float), dl.dataset._backup_ys, rcond=None)
        rs = A @ beta - dl.dataset._backup_ys
        dl.dataset.ys = rs
        # check if its parent and ancestors become invalid
        parent = node.parent
        while True:
            parent_valid_ops = get_valid_ops_pre_select(parent.path, parent.candidates, parent.max_depth, popped_paths)
            if len(parent_valid_ops) == 0:
                popped_paths.add(repr_path(*parent.path))
                parent = parent.parent
            else:
                break
    dl.dataset.ys = dl.dataset._backup_ys  # restore
    stats = pd.DataFrame(A, columns=header, index=dl.dataset.index)
    return results, stats