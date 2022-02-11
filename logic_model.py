import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class RuleModel(nn.Module):
    
    def __init__(self, input_dim, n_rules_list, nonnegative=False, skip_connect=True):
        super().__init__()
        print('Init Model:', n_rules_list)
        self.nonnegative = nonnegative
        self.logic_model = LogicModel(input_dim, n_rules_list, skip_connect)
        self.linear_layer = nn.Linear(n_rules_list[-1] * 2, 1, bias=False)
        if self.nonnegative != 'none':
            nn.init.uniform_(self.linear_layer.weight.data)
    
    def forward(self, x, tau):
        if not self.training:
            tau = 1e-9
        h = self.logic_model(x, tau)
        w = self.linear_layer.weight
        if self.nonnegative == 'relu':
            w = F.relu(w)
        elif self.nonnegative == 'softplus':
            w = F.softplus(w)
        h = h @ w.T
        return h.squeeze(-1)
    
    def get_rules(self, input_meaning):
        weights = self.linear_layer.weight.data[0, :]
        if self.nonnegative == 'relu':
            weights = F.relu(weights)
        elif self.nonnegative == 'softplus':
            weights = F.softplus(weights)
        weights = weights.cpu().numpy()
        rules = self.logic_model.get_rules(input_meaning)
        idx = np.argsort(-np.abs(weights))
        return [f'{weights[i]:.4e} x {rules[i]}' for i in idx]


class LogicModel(nn.Module):
    
    def __init__(self, input_dim, n_rules_list, skip_connect=True):
        super().__init__()
        self.input_layer = InputAugmentor()
        layers = []
        hidden_dim = input_dim * 2 + 2
        for n in n_rules_list[:-1]:
            layers.append(LogicLayer(hidden_dim, n, skip_connect=skip_connect))
            hidden_dim = layers[-1].output_dim
        layers.append(LogicLayer(hidden_dim, n_rules_list[-1], skip_connect=False))
        self.logic_layers = nn.ModuleList(layers)
    
    def forward(self, x, tau=1, verbose=False):
        if verbose:
            print(x.size())
        h = self.input_layer(x)
        if verbose:
            print(f'After InputLayer', h.size())
            print(h[0])
        for i, fn in enumerate(self.logic_layers):
            h = fn(h, tau)
            if verbose:
                print(f'After LogicLayer #{i}', h.size())
                print(h[0])
        return h
    
    def get_rules(self, input_meaning):
        h = self.input_layer.get_rules(input_meaning)
        for layer in self.logic_layers:
            h = layer.get_rules(h)
        return h


class InputAugmentor(nn.Module):
    
    def forward(self, x):
        assert x.dim() == 2
        return torch.cat([x, 1-x, torch.ones_like(x[:, :1]), torch.zeros_like(x[:, :1])], -1)
    
    def get_rules(self, input_rules):
        return input_rules + [f'(not {x})' for x in input_rules] + ['true', 'false']
    
    
class LogicLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, skip_connect=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2 + input_dim * skip_connect
        self.skip_connect = skip_connect
        self.conj = Conjunction(input_dim, output_dim)
        self.disj = Disjunction(input_dim, output_dim)
        
    def forward(self, x, tau=1):
        h1 = self.conj(x, tau)
        h2 = self.disj(x, tau)
        if self.skip_connect:
            return torch.cat([x, h1, h2], -1)
        else:
            return torch.cat([h1, h2], -1)
    
    def get_rules(self, input_rules):
        if self.skip_connect:
            return input_rules + self.conj.get_rules(input_rules) + self.disj.get_rules(input_rules)
        else:
            return self.conj.get_rules(input_rules) + self.disj.get_rules(input_rules)
        
        
class Conjunction(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight2 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        
        nn.init.normal_(self.weight1, std=1)
        nn.init.normal_(self.weight2, std=1)
        
    def forward(self, x, tau=1):
        # x: [batch_size, m]
        index1 = gumbel_softmax(self.weight1, tau=tau, hard=True, dim=0)
        index2 = gumbel_softmax(self.weight2, tau=tau, hard=True, dim=0)
        h1 = x @ index1  # [batch_size, n]
        h2 = x @ index2
        return h1 * h2
    
    def get_rules(self, input_rules):
        index1 = self.weight1.argmax(0).detach().cpu().numpy()
        index2 = self.weight2.argmax(0).detach().cpu().numpy()
        return [f'({input_rules[i]} and {input_rules[j]})' for i, j in zip(index1, index2)]
    
class Disjunction(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight2 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        
        nn.init.normal_(self.weight1, std=1)
        nn.init.normal_(self.weight2, std=1)
        
    def forward(self, x, tau=1):
        # x: [batch_size, m]
        index1 = gumbel_softmax(self.weight1, tau=tau, hard=True, dim=0)
        index2 = gumbel_softmax(self.weight2, tau=tau, hard=True, dim=0)
        h1 = x @ index1  # [batch_size, n]
        h2 = x @ index2
        return 1 - (1-h1) * (1-h2)

    def get_rules(self, input_rules):
        index1 = self.weight1.argmax(0).detach().cpu().numpy()
        index2 = self.weight2.argmax(0).detach().cpu().numpy()
        return [f'({input_rules[i]} or {input_rules[j]})' for i, j in zip(index1, index2)]


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

    
    
class MarginLoss(nn.Module):
    
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, pred_y, true_y):
        n = pred_y.size(0)
        indices = torch.triu_indices(n, n, 1)
        mask = (true_y.unsqueeze(0) > true_y.unsqueeze(1)).float()
        diff = pred_y.unsqueeze(0) - pred_y.unsqueeze(1)
        mask = mask[indices.unbind()]
        diff = diff[indices.unbind()]
        losses = torch.relu(self.margin - (mask * 2 - 1) * diff)
        loss = losses.mean()
        return loss

    
class BPRLoss(nn.Module):
    
    def forward(self, pred_y, true_y):
        n = pred_y.size(0)
        mask = (true_y.unsqueeze(0) > true_y.unsqueeze(1)).float()
        diff = pred_y.unsqueeze(0) - pred_y.unsqueeze(1)
        indices = torch.triu_indices(n, n, 1)
        mask = mask[indices.unbind()]
        diff = diff[indices.unbind()]
        loss = F.binary_cross_entropy_with_logits(diff, mask)
        return loss
        