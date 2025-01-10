from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge
import dgl
import torch.nn.functional as F

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            k_hop: int = 1,
            num_decoder_layers: int = 2,
            num_classes: int = 7,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._khop = k_hop
        self._num_decoder_layers = num_decoder_layers

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.encoder_label = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.decoder_label = setup_module(
            m_type=decoder_type,
            enc_dec="label_prediction",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=num_classes,
            num_layers=num_decoder_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )


        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
            self.encoder_to_decoder_label = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
            self.encoder_to_decoder_label = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.CrossEntropyLoss = F.nll_loss

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x, y):
        # ---- attribute reconstruction ----
        loss = self.combine_prediction(g, x, y)
        # loss = self.label_prediction(g,x,y,mask_label_nodes)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def get_label_prediction(self,g,x):
        with torch.no_grad():
            pre_use_g = g.clone()
            use_x = x.clone()

            
            use_g = pre_use_g

            enc_rep, all_hidden = self.encoder_label(use_g, use_x, return_hidden=True)
            if self._concat_hidden:
                enc_rep = torch.cat(all_hidden, dim=1)

            # ---- attribute reconstruction ----
            rep = self.encoder_to_decoder_label(enc_rep)

            if self._decoder_type in ("mlp", "liear") :
                recon = self.decoder_label(rep)
            else:
                recon = self.decoder_label(pre_use_g, rep)

        return recon
       
    def k_hop_subgraph(self,graph, node_ids, k):
        # Use DGL's built-in function to extract k-hop neighborhood
        sg_nodes, sg_edges = dgl.khop_in_subgraph(graph, node_ids, k)
        return sg_nodes, sg_edges
    
    def encoding_mask_noise_single(self, g, x, mask_node):
        use_g = g.clone()
        out_x = x.clone()
        num_nodes = g.num_nodes()
        total = list(range(num_nodes))
        total.remove(mask_node)
        keep_nodes = torch.tensor(total,device=x.device)
        subgraph_nodes, subgraph_edges = self.k_hop_subgraph(use_g, [mask_node], self._khop)
        # print(f"Subgraph nodes: {subgraph_nodes}")
        node_ids = subgraph_nodes.ndata[dgl.NID]
        # if mask_node >= 2708:
        #     print(node_ids)
        out_x[node_ids] = 0.0
        token_nodes = node_ids
        out_x[token_nodes] += self.enc_mask_token
        # node_id = 1

        # out_edges = use_g.out_edges(mask_node)
        # in_edges = use_g.in_edges(mask_node)


        # edges_to_remove = torch.cat([out_edges[0], in_edges[0]])

        # g_removed = dgl.remove_edges(use_g, edges_to_remove)
        return use_g, out_x, (node_ids, keep_nodes)

    def calculate_node_loss(self, g, x,mask_node, ori_prediction):
        # ---- single node prediction----

        with torch.no_grad():
            use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise_single(g, x,mask_node)
            enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)

            if self._concat_hidden:
                enc_rep = torch.cat(all_hidden, dim=1)
            rep = self.encoder_to_decoder(enc_rep)

            if self._decoder_type not in ("mlp", "linear"):
                # * remask, re-mask
                rep[mask_nodes] = 0

            if self._decoder_type in ("mlp", "linear") :
                recon = self.decoder(rep)
            else:
                recon = self.decoder(use_g, rep)
            x_init = x[mask_nodes]
            x_rec = recon[mask_nodes]

            loss = self.criterion(x_rec, x_init)

            ## label  prediction ##

            label_enc_rep, label_all_hidden = self.encoder_label(use_g, use_x, return_hidden=True)
            if self._concat_hidden:
                label_enc_rep = torch.cat(label_all_hidden, dim=1)
            label_rep = self.encoder_to_decoder_label(label_enc_rep)

            if self._decoder_type not in ("mlp", "linear"):
                # * remask, re-mask
                label_rep[mask_nodes] = 0

            if self._decoder_type in ("mlp", "linear") :
                label_recon = self.decoder_label(label_rep)
            else:
                label_recon = self.decoder_label(use_g, label_rep)
            subgraph_nodes, subgraph_edges = dgl.khop_out_subgraph(use_g, mask_node, self._khop+1)
            node_ids = subgraph_nodes.ndata[dgl.NID]
            node_ids = node_ids[node_ids != mask_node]

            y_rec = torch.exp(label_recon[node_ids]) + 1e-45
            y_init = ori_prediction[node_ids] 
            m = 0.5 * (y_rec + y_init)

            loss_label = F.kl_div(y_rec.log(), y_init, reduction='none') +  F.kl_div(y_rec.log(), m, reduction='none') + F.kl_div(y_init.log(), m, reduction='none')
            loss_label = torch.mean(loss_label, dim=-1)
            for i in range(loss_label.size(0)):
                if torch.isnan(loss_label[i]):
                    loss_label[i] = torch.zeros_like(loss) 
                if torch.isinf(loss_label[i]):
                    loss_label[i] = torch.ones_like(loss) * 10000000
            loss_label = torch.mean(loss_label)



            # loss_elements = F.kl_div(y_rec.log(), y_init, reduction='none')

            # loss_label = torch.mean(loss_elements)
            # return loss_label
                # exit()
            
            loss_item = {"attr_loss": loss.item(), "label_loss": loss_label.item()}
            
        return loss, loss_item

    def calculate_label_deviation(self, g,x, y,ori_prediction, mask_node):
        with torch.no_grad():
            use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise_single(g, x,mask_node)
            subgraph_nodes, subgraph_edges = dgl.khop_in_subgraph(use_g, mask_node, 4)

            return g.subgraph(subgraph_nodes.ndata[dgl.NID])
            enc_rep, all_hidden = self.encoder_label(use_g, use_x, return_hidden=True)
            if self._concat_hidden:
                enc_rep = torch.cat(all_hidden, dim=1)
            rep = self.encoder_to_decoder(enc_rep)

            if self._decoder_type not in ("mlp", "linear"):
                # * remask, re-mask
                rep[mask_nodes] = 0

            if self._decoder_type in ("mlp", "linear") :
                recon = self.decoder(rep)
            else:
                recon = self.decoder_label(use_g, rep)
            subgraph_nodes, subgraph_edges = self.k_hop_subgraph(use_g, mask_node, self._khop+1)
            node_ids = subgraph_nodes.ndata[dgl.NID]
            node_ids = node_ids[node_ids != mask_node]

            y_rec = torch.exp(recon[node_ids])
            y_init = ori_prediction[node_ids]
            
            loss = F.kl_div(y_rec.log(), y_init, reduction='mean')
            if torch.isnan(loss):
                print(y_rec, y_init, loss)
                exit()
                loss = torch.zeros_like(loss) - 1
            if torch.isinf(loss):
                print(y_rec, y_init, loss)
                exit()
            loss_item = {"loss": loss.item()}
        # encoder_params_after = {name: param.clone() for name, param in self.encoder.named_parameters()}
        # for name in encoder_params_before:
        #     diff = torch.norm(encoder_params_after[name] - encoder_params_before[name]).item()
        #     assert diff == 0

        return loss, loss_item
    
    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss
    
    def label_prediction(self, g, x, y,mask_label_nodes):
        pre_use_g = g.clone()
        use_x = x.clone()

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)


        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder_label(pre_use_g, rep)

        y_init = y[mask_label_nodes]
        y_rec = recon[mask_label_nodes]
        loss = F.nll_loss(y_rec, y_init)
        return loss

    def combine_prediction(self, g, x, y):
        label_g = g.clone()
        label_x = x.clone()
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)


        # ---- label reconstruction ----
        label_enc_rep, label_all_hidden = self.encoder_label(label_g, label_x, return_hidden=True)
        if self._concat_hidden:
            label_enc_rep = torch.cat(label_all_hidden, dim=1)

        label_rep = self.encoder_to_decoder_label(label_enc_rep)


        if self._decoder_type in ("mlp", "liear") :
            label_recon = self.decoder_label(label_rep)
        else:
            label_recon = self.decoder_label(label_g, label_rep)

        y_init = y[mask_nodes]
        y_rec = label_recon[mask_nodes]
        label_loss = F.nll_loss(y_rec, y_init)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        attr_loss = self.criterion(x_rec, x_init)
        loss = label_loss + attr_loss
        return loss

    def embed(self, g, x):
        rep, all_hidden= self.encoder(g, x, return_hidden=True)
        return rep,all_hidden
    
    def embed_for_label_prediction(self, g, x):
        rep, all_hidden= self.encoder_label(g, x, return_hidden=True)
        return rep,all_hidden
    
    def generated_masked_node_rep(self, g, inputs, number_layers):
        rep = self.encoder.forward_only_necessary(g, inputs, number_layers)
        
        return rep
    
    def generated_masked_node_rep_for_label_prediction(self, g, inputs, number_layers):
        rep = self.encoder_label.forward_only_necessary(g, inputs, number_layers)
        
        return rep


    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
