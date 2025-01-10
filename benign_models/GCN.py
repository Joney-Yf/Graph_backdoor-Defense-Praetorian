#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1)
    def get_h(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, idx_attach = None, trigger_nodes = None, adv_unlearn=False, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            if adv_unlearn:
                self.adv_unlearn(self.labels, idx_train, idx_val, idx_attach, trigger_nodes, train_iters, verbose)
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # torch.cuda.empty_cache()

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()
    
    def adv_unlearn(self, labels, idx_train, idx_val, idx_attach, trigger_nodes, train_iters, verbose=False, epsilon=0.05):
        
        idx_train_set = set(idx_train.cpu().tolist())
        idx_attach_set = set(idx_attach)
        idx1 = torch.tensor(list(idx_train_set - idx_attach_set), dtype=torch.long, device=self.device)
        idx2 = torch.tensor(idx_attach, dtype=torch.long, device=self.device)
        trigger_nodes = torch.tensor(trigger_nodes, dtype=torch.long, device=self.device)
        idx2 = torch.cat([idx2, trigger_nodes])
        # print(idx2.shape)


        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        poison_edge_index = self.edge_index.clone()
        poison_edge_weights = self.edge_weight.clone()

        mask = torch.zeros(len(self.features), dtype=torch.bool, device=self.features.device)
        mask[trigger_nodes] = 1
        keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
        # Filter the edge_index by the edges we want to keep
        filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
        # Filter the edge weights similarly
        filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            
            # mask trigger node
            

            output = self.forward(self.features, filtered_poison_edge_index, filtered_poison_edge_weights)
            

            loss_train_main = F.nll_loss(output[idx1], labels[idx1])
            

            # victim_output = output[idx2]
            # uniform_dist = torch.ones_like(victim_output) / victim_output.size(1)
            # loss_victim_reg = F.kl_div(F.log_softmax(victim_output, dim=1), 
            #                         uniform_dist, 
            #                         reduction='batchmean')
            output2 = self.forward(self.features, self.edge_index, self.edge_weight)
            probs = F.softmax(output2[idx2], dim=1)
            target_probs = probs[range(len(idx2)), torch.tensor([labels[idx2[0]]]*len(idx2), device=idx2.device)]
            loss_victim_reg = torch.mean(target_probs)  
            

            # self.features.requires_grad = True
            # output_adv = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_adv = F.nll_loss(output_adv[idx1], labels[idx1])
            # loss_adv.backward(retain_graph=True) 
            
            # grad = self.features.grad.clone().detach()
            # self.features.requires_grad = False
            
            # perturb = torch.zeros_like(self.features)
            # perturb[trigger_nodes] = epsilon * grad[trigger_nodes].sign()
            
            # output_perturbed = self.forward(self.features + perturb, self.edge_index, self.edge_weight)
            # loss_adv_perturb = F.nll_loss(output_perturbed[idx1], labels[idx1])
            

            total_loss = loss_train_main +0.75 * loss_victim_reg 
            

            # optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if verbose and (i+1) % 10 == 0:
                with torch.no_grad():
                    out_val = self.forward(self.features, self.edge_index, self.edge_weight)
                    val_loss = F.nll_loss(out_val[idx_val], labels[idx_val])
                    val_acc = (out_val[idx_val].argmax(dim=1) == labels[idx_val]).float().mean().item()
                print(f"Epoch {i+1}: total_loss={total_loss.item():.4f}, val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}")
        
        self.eval()
        self.output = output



    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # torch.cuda.empty_cache()


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids

# %%
