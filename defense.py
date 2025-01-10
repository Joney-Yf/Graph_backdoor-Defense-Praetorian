import logging
import numpy as np
from tqdm import tqdm
import torch
import dgl.function as fn
from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.models import build_model
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
import json
import dgl
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
import time
def pretrain(model, graph, feat, label,optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    y = label.to(device)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f} current lr is: {get_current_lr(optimizer):.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        # if (epoch + 1) % 200 == 0:
        #     node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
    # return best_model
    return model

def victim_node_selection(old_g, x, original_len, all_loss, attack_nodes, train_idx, labels, isolated_nodes):
    alpha = 1
    max_iters = 2  #
    g = old_g.clone()
    all_loss = torch.tensor(list(all_loss.values()),device = g.device).pow(9)
    g.ndata['feat'] = all_loss
    g = g.remove_self_loop()
    all_src, all_dst = g.edges()
    train_idx = train_idx.to(g.device)
    attack_nodes = attack_nodes.to(g.device)
    train_idx = torch.cat([train_idx, attack_nodes])
    # edges_to_remove_mask = (torch.isin(all_src, train_idx)) & (torch.isin(all_dst, train_idx))
    edges_to_remove_mask = (torch.isin(all_src, train_idx))
    edges_to_remove = torch.nonzero(edges_to_remove_mask, as_tuple=False).squeeze()
    if edges_to_remove.numel() > 0:
        g.remove_edges(edges_to_remove)
    g = g.add_self_loop()
    A0 =  all_loss.clone()
    for i in range(max_iters):
        g.update_all(fn.copy_u('feat','m'), fn.mean('m','mean_feat'))
        mean_feat = g.ndata['mean_feat']
        new_feat = alpha * mean_feat +  A0
        g.ndata['feat'] = new_feat
        
    all_loss = {}
    train_idx = train_idx.tolist()
    attack_nodes = attack_nodes.tolist()
    # src, dst= g.in_edges(1556)
    # exit()
    print('the total len is ', len(train_idx))
    for i, value in enumerate(g.ndata['feat']):
        if i in train_idx:
            all_loss[i] = value.item() 
    sorted_nodes = sorted(all_loss.items(), key=lambda x: x[1], reverse=True)
    attack = {}
    for i in range(original_len,original_len+10000):
        attack[i] = 0
    rank = []
    selected_nodes = []
    failed_count = 0
    print("node\taverage")
    target_label = 0
    threshold = 0
    for i, (node, avg_loss) in tqdm(enumerate(sorted_nodes)):
        if i == 0:
            target_label = labels[node].item()
            selected_nodes.append(node)
        else:
            if labels[node].item() == target_label:
                selected_nodes.append(node) 
                failed_count = 0
                threshold = 2 * avg_loss
            else:
                failed_count += 1
            
            if failed_count > 1:
                break
        if node in attack_nodes:
            print(f"rank is {i}, and {node}\t{avg_loss:.8f} and label is \t{labels[node].item()}")
            rank.append(i)
    if len(rank) == 0:
        print('TP is zero')
    else:
        print('average rank is', sum(rank)/len(rank))


    TP=0.0
    FP=0.0
    TN=0.0
    for node in selected_nodes:
        if node in attack_nodes:
            TP+=1.0
        else:
            FP+=1.0
    FN = float(len(attack_nodes) - TP)
    if TP==0:
        pre = 0
    else:
        pre = TP/(TP+FP)
    rec = TP/(TP+FN)

    print('victim precision is ', pre)
    print('victim recall is ', rec)

    subgraph_nodes, subgraph_edges = dgl.khop_in_subgraph(g, selected_nodes, 2)
    sus_triggers = subgraph_nodes.ndata[dgl.NID]
    trigger_selection = []
    for node in sus_triggers:
        if g.ndata['feat'][node] > threshold and (node.item() not in selected_nodes):
            trigger_selection.append(node)
    TP=0.0
    FP=0.0
    TN=0.0
    # print(trigger_selection)
    for node in trigger_selection:
        if node >= original_len:
            TP+=1.0
        else:
            FP+=1.0
    iso_num = 0
    for node in range(original_len, len(g.ndata['feat'])):
        if node in isolated_nodes:
            iso_num += 1
    FN = float(len(g.ndata['feat']) - original_len - iso_num - TP)
    if TP==0:
        pre = 0
    else:
        pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    
    print('trigger precision is ', pre)
    print('trigger recall is ', rec)

    return selected_nodes, trigger_selection,target_label

def node_rank(old_g, x,original_len, all_loss, attack_nodes, train_idx):
    sorted_nodes = sorted(all_loss.items(), key=lambda x: x[1], reverse=True)
    attack = {}
    for i in range(original_len,original_len+10000):
        attack[i] = 0
    rank = []
    selected_nodes = []
    failed_count = 0
    print("node\taverage")
    for i, (node, avg_loss) in tqdm(enumerate(sorted_nodes)):
        if node in attack:
            print(f"rank is {i}, and {node}\t{avg_loss:.8f}")
            rank.append(i)
    if len(rank) > 0:
        print('average rank is', sum(rank)/len(rank))

def victim_and_trigger_nodes_detection(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, benign_graph, (num_features, num_classes), original_len, attack_nodes, train_idx = load_dataset(dataset_name,args.backdoor_train_file)
    args.num_features = num_features
    args.num_classes = num_classes
    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
        x = graph.ndata["feat"]
        y = graph.ndata["label"]
        if not load_model:
            model = pretrain(model, graph, x, y, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()
        x = x.to(device)
        graph = graph.to(device)
        y = graph.ndata['label'].to(device)
        attr_loss_valid = {}
        label_loss_valid = {}
        tmp_g = graph.clone()
        in_degrees = tmp_g.in_degrees()
        out_degrees = tmp_g.out_degrees() 
        save_loss_dict = {}
        isolated = (in_degrees == 1) & (out_degrees == 1)
        isolated_nodes = set(torch.nonzero(isolated).squeeze().tolist())
        if isinstance(isolated_nodes, int):
            isolated_nodes = set([isolated_nodes])
        with torch.no_grad():
            ori_prediction = torch.exp(model.get_label_prediction(graph,x))
            print(torch.exp(model.get_label_prediction(graph,x)[attack_nodes]))
            for i in tqdm(range(len(x))):
                if i in isolated_nodes:
                    continue
                else:
                    loss, loss_dict = model.calculate_node_loss(graph, x,i, ori_prediction)
                in_d = in_degrees[i].item()
                out_d = out_degrees[i].item()
                if in_d > 0:
                    attr_loss_valid[i] = 1.0 / (in_degrees[i].item() * loss_dict['attr_loss'])
                else:
                    attr_loss_valid[i] = 1.0 / loss_dict['attr_loss']
                if out_d>0:
                    label_loss_valid[i] = loss_dict['label_loss']/out_degrees[i].item()
                else:
                    label_loss_valid[i] = loss_dict['label_loss']
                save_loss_dict[i] = loss_dict


            file_name = args.backdoor_train_file.split('/')[-1]
            with open('loss_dict_{}.json'.format(file_name), 'w', encoding='utf-8') as f:
                json.dump(save_loss_dict, f, ensure_ascii=False, indent=4)
            

        
        attr_loss = torch.tensor(list(attr_loss_valid.values()), device = x.device)
        attr_std = attr_loss.std()
        attr_mean = attr_loss.mean()
        print(attr_mean)
        print(attr_std)
        attr_loss = (attr_loss - attr_mean) / attr_std
        label_loss = torch.tensor(list(label_loss_valid.values()), device = x.device)
        label_mean = label_loss.mean()
        label_std = label_loss.std()
        print(label_mean )
        print(label_std)
        label_loss = (label_loss - label_mean) / label_std

        '''
        def bimodality_coefficient(data):
            data = np.asarray(data)
            n = len(data)
            s = skew(data)
            k = kurtosis(data, fisher=True)  # fisher=True =>0

            if n < 4:
                return np.nan

            bc = (s**2 + 1) / (k + 3 * (n-1)**2 / ((n-2)*(n-3)) + 1e-9)
            return bc

        attr_bc_value = bimodality_coefficient(np.array(attr_loss.cpu())) 
        label_bc_value = bimodality_coefficient(np.array(label_loss.cpu())) 

        alpha= 50
        mean_v =  (attr_bc_value + label_bc_value) / 2.0
        delta = attr_bc_value - label_bc_value
        sig = alpha * (mean_v**3) * delta
        w_i = torch.sigmoid(torch.tensor(sig))
        w_e = 1 - w_i
        # intra_correlation = torch.exp(torch.tensor(attr_bc_value, device=graph.device))
        # eternal_influence = torch.exp(torch.tensor(label_bc_value, device=graph.device)) 
        # intra_correlation = intra_correlation/eternal_influence
        print('attr_loss bc value is', attr_bc_value)
        print('label_loss bc value is', label_bc_value)

        print('weight is', w_i, w_e)
        '''

        node_with_loss = {}
        attr_loss_dict = {}
        label_loss_dict = {}
        # label_with_loss = {}
        for i in tqdm(range(len(x))):
            if i in isolated_nodes:
                attr_loss_valid[i] = 0
                label_loss_valid[i]= 0
            else:
                attr_loss_valid[i] = (attr_loss_valid[i] - attr_mean)/attr_std
                label_loss_valid[i] = (label_loss_valid[i] - label_mean)/label_std
            # node_with_loss[i] =  w_i * attr_loss_valid[i] +  w_e * label_loss_valid[i] 

            attr_loss_dict[i] = attr_loss_valid[i]
            label_loss_dict[i] = label_loss_valid[i]
        print('########################## attr loss rank #################################')
        node_rank(graph, x,original_len,attr_loss_dict, attack_nodes,train_idx)
        selected_nodes_attr, trigger_selection_attr, target_label_attr = victim_node_selection(graph, x,original_len,attr_loss_dict, attack_nodes, train_idx, y, isolated_nodes)
        print('########################## label loss rank #################################')
        node_rank(graph, x,original_len,label_loss_dict, attack_nodes,train_idx)
        selected_nodes_label, trigger_selection_label, target_label_label = victim_node_selection(graph, x,original_len,label_loss_dict, attack_nodes, train_idx, y, isolated_nodes)
        print('########################## overall loss rank #################################')

        w_i = float(len(selected_nodes_attr)) * float(len(trigger_selection_attr))
        w_e = float(len(selected_nodes_label)) * float(len(trigger_selection_label))
        if w_e == 0 and w_i == 0:
            w_i = 1.0 
            w_e = 1.0
        elif w_e ==0:
            w_e = 1.0
        
        if target_label_attr == target_label_label:
            w_i = w_i / w_e
        else:
            if w_i < w_e:
                w_i = w_i / 10
            elif w_i > w_e:
                w_i = w_i * 10
            
            w_i = w_i / w_e
        w_i = min(w_i, 100)
        w_i = max(0.01, w_i)
        print("weights are", w_i,  w_e)
        for i in tqdm(range(len(x))):
            node_with_loss[i] =  w_i * attr_loss_valid[i] +  label_loss_valid[i] 
        node_rank(graph, x,original_len,node_with_loss, attack_nodes,train_idx)
        selected_nodes, trigger_selection, target_label_overall = victim_node_selection(graph, x,original_len,node_with_loss, attack_nodes, train_idx, y, isolated_nodes)
        # print('########################## victim node selection #################################')
        # selected_nodes, trigger_selection = victim_node_selection(graph, x,original_len,node_with_loss, attack_nodes, train_idx, y, isolated_nodes)
    
    return selected_nodes,trigger_selection, benign_graph

    
def defense(args):
    from benign_models.construct import model_construct

    backdoor_train_file = args.backdoor_train_file
    backdoor_test_file = backdoor_train_file.replace('train', 'test')
    if 'DPGBA' in backdoor_train_file:
        args.target_class=2


    victim_nodes, trigger_selection, benign_graph = victim_and_trigger_nodes_detection(args)



    print('########## Before Defense, the Attack Performance #############')
    x, full_edge_index, edge_index, edge_weight,labels, train_idx, val_idx, attack = torch.load(backdoor_train_file, map_location=torch.device(args.device))
    test_model = model_construct(args,args.test_model,x, labels,args.device).to(args.device) 
    bkd_tn_nodes = torch.cat([train_idx,attack]).to(args.device)
    test_model.fit(x, edge_index, edge_weight, labels, bkd_tn_nodes, val_idx,train_iters=400,verbose=False)

    induct_x, induct_edge_index, induct_edge_index, induct_edge_weight, labels, train_idx, val_idx, idx_atk, idx_clean_test = torch.load(backdoor_test_file, map_location=torch.device(args.device))
    induct_x, induct_edge_index,induct_edge_weight = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weight.clone().detach()
    output = test_model(induct_x, induct_edge_index, induct_edge_weight)    
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    asr = train_attach_rate
    flip_idx_atk = idx_atk[(benign_graph.ndata['label'][idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    ca = test_model.test(induct_x,induct_edge_index,induct_edge_weight,benign_graph.ndata['label'],idx_clean_test)
    print('The ASR is {}\nThe CA is {}'.format(flip_asr, ca))


    print('########## After Defense, the Attack Performance #############')
    test_model = model_construct(args,args.test_model,x, labels,args.device).to(args.device) 
    bkd_tn_nodes = torch.cat([train_idx,attack]).to(args.device)
    test_model.fit(x, edge_index, edge_weight, labels, bkd_tn_nodes, val_idx, victim_nodes, trigger_selection, adv_unlearn=True, train_iters=400,verbose=False)
    induct_x, induct_edge_index, induct_edge_index, induct_edge_weight, labels, train_idx, val_idx, idx_atk, idx_clean_test = torch.load(backdoor_test_file, map_location=torch.device(args.device))
    induct_x, induct_edge_index,induct_edge_weight = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weight.clone().detach()
    trigger_selection = torch.tensor(trigger_selection, dtype=torch.long, device=induct_x.device)
    mask = torch.zeros(len(induct_x), dtype=torch.bool, device=induct_x.device)
    mask[trigger_selection] = 1
    keep_edges_mask = ~(mask[induct_edge_index[0]] | mask[induct_edge_index[1]])
    # Filter the edge_index by the edges we want to keep
    filtered_poison_edge_index = induct_edge_index[:, keep_edges_mask]
    # Filter the edge weights similarly
    filtered_poison_edge_weights = induct_edge_weight[keep_edges_mask]

    output = test_model(induct_x, filtered_poison_edge_index, filtered_poison_edge_weights)    
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    asr = train_attach_rate
    flip_idx_atk = idx_atk[(benign_graph.ndata['label'][idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    ca = test_model.test(induct_x,filtered_poison_edge_index,filtered_poison_edge_weights,benign_graph.ndata['label'],idx_clean_test)
    print('The ASR is {}\nThe CA is {}'.format(flip_asr, ca))




        


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    defense(args)
