from search import *
from dataset import *
import time
from argparse import ArgumentParser


def run(max_depth, topk, iterations, batch_size, folder, trainfile, validfile=None, testfile=None, memory=False, logprob=False, cpucore=1, prefix='', out_prefix=''):
    path = pathlib.Path(folder)
    if out_prefix == '':
        out_prefix = prefix
    if len(out_prefix):
        outpath = path / f'{out_prefix}_stats'
    else:
        outpath = path / 'stats'
    if len(prefix):
        trainfile = f'{prefix}_{trainfile}'
        if validfile is not None:
            validfile = f'{prefix}_{validfile}'
        if testfile is not None:
            testfile = f'{prefix}_{testfile}'
    outpath.mkdir(exist_ok=True)
    # Training
    train_ds = RecordDataset(predfile=trainfile, prefix_path=path, memory=memory, logprob=logprob)
    train_dl = RecordDataloader(train_ds, batch_size=batch_size)
    ops = init_ops(train_ds)
    # nodes = get_topk_stats(ops, max_depth, train_dl, iterations, topk)
    nodes, stats = get_topk_stats_sequential(ops, max_depth, train_dl, iterations, topk, cpucore)
    for n in nodes: print(n)
    # Save stats meanings
    with open(outpath / 'stats_meanings.txt', 'w') as fh:
        fh.write('\n'.join([repr_path(*n.path) for n in nodes]))
        fh.write('\n')
    # Encode and Save
    stats.to_csv(outpath / 'train_stats.csv', index=True, header=True)
    header = stats.columns
    if validfile and (path / validfile).exists():
        valid_ds = RecordDataset(predfile=validfile, prefix_path=path, memory=memory)
        encode_files(nodes, valid_ds, header, outpath / 'valid_stats.csv', cpucore)
    if testfile and (path / testfile).exists():
        test_ds = RecordDataset(predfile=testfile, prefix_path=path, memory=memory)
        encode_files(nodes, test_ds, header, outpath / 'test_stats.csv', cpucore)
        

def encode_files(nodes, dataset, header, fname, cpucore):
    stats = []
    for i, node in enumerate(tqdm.tqdm(nodes, desc='encoding')):
        current_stats = pipe_pre_select_parallel(node.path, (x[0] for x in dataset), cpucore)
        current_stats = np.asarray([s.values if isinstance(s, pd.Series) else s for s in current_stats])
        current_stats = np.nan_to_num(current_stats).reshape(len(dataset), -1)
        stats.append(current_stats)
    A = np.concatenate(stats, 1)
    stats = pd.DataFrame(A, columns=header, index=dataset.index)
    stats.to_csv(fname, index=True, header=True)

            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--trainfile', type=str, default='train_preds.csv')
    parser.add_argument('--validfile', type=str, default='valid_preds.csv')
    parser.add_argument('--testfile', type=str, default='test_preds.csv')
    parser.add_argument('--inmemory', action='store_true')
    parser.add_argument('--logprob', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpucore', type=int, default=4)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--outprefix', type=str, default='')
    args = parser.parse_args()
    args.depth += 1  # to match the definitions in the paper
    np.random.seed(args.seed)
    t0 = time.time()
    run(args.depth, args.topk, args.iterations, args.bs, args.folder, args.trainfile, args.validfile, args.testfile, memory=args.inmemory, logprob=args.logprob, cpucore=args.cpucore, prefix=args.prefix, out_prefix=args.outprefix)
    delta_t = time.time() - t0
    print(f'Elpased Time: {delta_t:.2f}s')