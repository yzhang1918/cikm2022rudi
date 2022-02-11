import numpy as np
import pandas as pd

import tqdm
import pathlib
from argparse import ArgumentParser
from scipy.stats import pearsonr, kendalltau

import torch
from torch import nn, optim
from logic_model import *
from torch.utils.data import TensorDataset, DataLoader


def load_files(path, prefix, name, onlytopk):
    if len(prefix):
        statpath = path / f'{prefix}_stats'
        fname = f'{prefix}_{name}_preds.csv'
    else:
        statpath = path / 'stats'
        fname = f'{name}_preds.csv'
    df = pd.read_csv(statpath / f'{name}_stats.csv', header=0, index_col=0)
    if onlytopk > 0:
        cols = [x for x in df.columns if int(x[1:].split('D')[0]) < onlytopk]
        df = df[cols]
    ys = pd.read_csv(path / fname, index_col=0, header=None).values.squeeze()
    return df, ys


def load_data(path, prefix, onlytopk):
    if len(prefix):
        statpath = path / f'{prefix}_stats'
    else:
        statpath = path / 'stats'
    # load stats meanings
    meaning_file = statpath / 'meanings.txt'
    if not meaning_file.exists():
        meaning_file = statpath / 'stats_meanings.txt'
    with open(meaning_file) as fh:
        meanings = fh.read().strip().split('\n')
    # load dfs
    train_df, train_ys = load_files(path, prefix, 'train', onlytopk)
    n_train = len(train_df)
    dfs = [train_df]
    use_valid = (statpath / 'valid_stats.csv').exists()
    use_test = (statpath / 'test_stats.csv').exists()
    if use_valid:
        valid_df, valid_ys = load_files(path, prefix, 'valid', onlytopk)
        n_valid = len(valid_df)
        dfs.append(valid_df)
    if use_test:
        test_df, test_ys = load_files(path, prefix, 'test', onlytopk)
        n_test = len(test_df)
        dfs.append(test_df)
    df_full = pd.concat(dfs)
    df_full.columns = [translate_name(n, meanings) for n in df_full.columns]
    binary_feat_df = encode_binary(df_full)
    print(f'Stats Shape: {df_full.shape}  Binary Feat Shape: {binary_feat_df.shape}')
    
    train_xs = binary_feat_df.iloc[:n_train].values
    train_ds = TensorDataset(torch.from_numpy(train_xs).float(), torch.from_numpy(train_ys).float())
    if use_valid:
        valid_xs = binary_feat_df.iloc[n_train:n_train+n_valid].values
        valid_ds = TensorDataset(torch.from_numpy(valid_xs).float(), torch.from_numpy(valid_ys).float())
    else:
        valid_ds = None
    if use_test:
        test_xs = binary_feat_df.iloc[-n_test:].values
        test_ds = TensorDataset(torch.from_numpy(test_xs).float(), torch.from_numpy(test_ys).float())
    else:
        test_ds = None
    return list(binary_feat_df.columns), train_ds, valid_ds, test_ds
    

def translate_name(name, meanings):
    # 'Rxx'
    # 'RxxDxx'
    if 'D' in name:
        j = name.index('D')
        i = int(name[1:j])
        return meanings[i] + f'[={name[j+1:]}]'
    else:
        i = int(name[1:])
        return meanings[i]


def encode_binary(df):
    df = df.fillna(0)
    binary_dfs = []
    n_percentiles = 10
    for col in df:
        x = df[col]
        if x.nunique() <= 2:
            unique_vals = set(x)
            if len(unique_vals - {0, 1}) == 0:
                binary_dfs.append(x)
                continue
        percentiles = np.percentile(x, np.linspace(0, 100, 1+n_percentiles))
        percentiles = np.unique(percentiles)
        compare_df = pd.DataFrame(x.values[:, None] > percentiles, columns=[f'{col} > {p:.4e}' for p in percentiles], index=df.index)
        binary_dfs.append(compare_df)
    binary_feat_df = pd.concat(binary_dfs, 1)
    return binary_feat_df.astype(float)


def eval_model(model, dl, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            h = model(x, tau=1e-9)
            preds.append(h.detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds

    
def train_rule_model(meanings, n_rules_list, loss_type, nonnegative, skip_connect,
                     lr0, lr1, tau0, tau1, n_epochs, batch_size, device, train_ds, valid_ds=None, test_ds=None):
    device = torch.device(device)
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'margin':
        loss_fn = MarginLoss()
    elif loss_type == 'bpr':
        loss_fn = BPRLoss()
    else:
        raise NotImplementedError
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if valid_ds is not None:
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = RuleModel(len(meanings), n_rules_list, nonnegative=nonnegative, skip_connect=skip_connect).to(device)
    optimizer = optim.Adam(model.parameters(), lr0)
    valid_preds = None
    test_preds = None
    for i in range(n_epochs):
        # Training
        lr = lr0 + (lr1 - lr0) * i / (n_epochs - 1)
        for g in optimizer.param_groups:
            g['lr'] = lr
        tau = tau0 + (tau1 - tau0) * i / (n_epochs - 1)
        cum_loss = 0.
        model.train()
        ys = []
        hs = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            h = model(x, tau)
            ys.append(y.cpu().numpy())
            hs.append(h.detach().cpu().numpy())
            loss = loss_fn(h, y)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        batch_loss = cum_loss / len(train_dl)
        ys = np.concatenate(ys)
        hs = np.concatenate(hs)
        rho_score = pearsonr(ys, hs)[0]
        tau_score = kendalltau(ys, hs)[0]
        print(f'Epoch {i+1:3d} LR={lr:.4f} Tau={tau:.4f} Loss={batch_loss:.4f} Rho={rho_score:.4f} Tau={tau_score:.4f}')
    del train_dl
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_preds = eval_model(model, train_dl, device)
    # Validation
    if valid_ds is not None:
        valid_preds = eval_model(model, valid_dl, device)
    # Testing
    if test_ds is not None:
        test_preds = eval_model(model, test_dl, device)
    return model.get_rules(meanings), train_preds, valid_preds, test_preds


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--outprefix', type=str, default='')
    parser.add_argument('--loss', type=str, default='bpr')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr0', type=float, default=0.1)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--tau0', type=float, default=1.)
    parser.add_argument('--tau1', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n_rules', type=int, default=20) 
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--nonnegative', type=str, default='none')
    parser.add_argument('--no_skip_connect', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--onlytopk', type=int, default=0)

    args = parser.parse_args()
    args.n_rules //= 2  # to match the definitions in the paper
    args.n_layers -= 1

    path = pathlib.Path(args.folder)
    args = parser.parse_args()
    seed_all(args.seed)

    n_rules_list = [args.hidden_size for _ in range(args.n_layers)] + [args.n_rules]

    meanings, train_ds, valid_ds, test_ds = load_data(path, args.prefix, args.onlytopk)
    rules, train_preds, valid_preds, test_preds = train_rule_model(
        meanings, n_rules_list=n_rules_list, loss_type=args.loss, nonnegative=args.nonnegative, 
        skip_connect=(not args.no_skip_connect),
        lr0=args.lr0, lr1=args.lr1, tau0=args.tau0, tau1=args.tau1, n_epochs=args.epochs, 
        batch_size=args.bs, device=args.device, 
        train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds)
    
    out_prefix = args.outprefix
    if out_prefix == '':
        out_prefix = args.prefix
                                                      
    if out_prefix:
        outpath = path / f'{out_prefix}-rudi_rules'
    else:
        outpath = path / 'rudi_rules'
    outpath.mkdir(exist_ok=True)
    with open(outpath / 'rules.txt', 'w') as fh:
        fh.write('\n'.join(rules))
    with open(outpath / 'rule_model_train_outputs.csv', 'w') as fh:
        fh.write('\n'.join([str(i) for i in train_preds]))
    if valid_preds is not None:
        with open(outpath / 'rule_model_valid_outputs.csv', 'w') as fh:
            fh.write('\n'.join([str(i) for i in valid_preds]))
    if test_preds is not None:
        with open(outpath / 'rule_model_test_outputs.csv', 'w') as fh:
            fh.write('\n'.join([str(i) for i in test_preds]))