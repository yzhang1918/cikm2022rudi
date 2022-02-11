import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score


def describe_predictions(y_true, pred_probs):
    auc = roc_auc_score(y_true, pred_probs)
    if pred_probs.min() >= 0 and pred_probs.max() <= 1:
        th = 0.5
    else:
        th = np.percentile(pred_probs, y_true.mean() * 100)
    p = precision_score(y_true, pred_probs>th)
    r = recall_score(y_true, pred_probs>th)
    acc = accuracy_score(y_true, pred_probs>th)
    s = f'AUC={auc:.4f} P={p:.4f} R={r:.4f} Acc={acc:.4f}'
    detail = dict(AUC=auc, P=p, R=r, Acc=acc)
    return s, detail


class VEWS:
    # User: 33576
    # lgbm_feat_dim = 300
    n_train = 25861
    n_valid = 1000
    n_test = 6715
    path = 'vews_all'
    cat_cols = 'type meta consecutive reversion threehop common fast'.split()
    vocab_sizes = [3, 2, 3, 3, 4, 4, 4]
    memory = False
    kwargs = dict(cat_cols=cat_cols, vocab_sizes=vocab_sizes, memory=memory)
    

class ELO:
    # User: 201917
    n_train = 160534
    n_valid = 1000
    n_test = 40383
    path = 'elo_10'
    cat_cols = 'authorized_flag city_id category_1 installments category_3 merchant_category_id merchant_id month_lag category_2 state_id subsector_id new'.split()
    vocab_sizes = [3, 11, 3, 11, 5, 11, 12, 11, 7, 11, 11, 2]
    num_cols = ['purchase_amount']
    sortby_col = 'purchase_date'
    kwargs = dict(cat_cols=cat_cols, vocab_sizes=vocab_sizes, num_cols=num_cols, sortby_col=sortby_col)


class RedHat:
    # User: 144639
    n_train = 114711
    n_valid = 1000
    n_test = 28928
    path = 'red_hat_10'
    cat_cols = 'activity_category char_1 char_2 char_3 char_4 char_5 char_6 char_7 char_8 char_9 char_10'.split()
    vocab_sizes = [8, 12, 12, 12, 9, 9, 7, 10, 12, 12, 12]
    sortby_col = 'date'
    kwargs = dict(cat_cols=cat_cols, vocab_sizes=vocab_sizes, sortby_col=sortby_col)

class Wiki:
    # User: 8227
    n_train = 5582
    n_valid = 1000
    n_test = 1645
    path = 'wiki'
    num_cols = [f'x{i}' for i in range(172)]
    kwargs = dict(num_cols=num_cols)

datasets = {
    'vews': VEWS,
    'elo': ELO,
    'redhat': RedHat,
    'redhatS': RedHatStatic,
    'wiki': Wiki,
}
