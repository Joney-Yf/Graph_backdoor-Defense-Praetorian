# %%
from benign_models.GCN import GCN

def model_construct(args,model_name,x,y,device):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    if (model_name == 'GCN'):

        model = GCN(nfeat=x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay_benign,\
                    device=device,
                    use_ln=use_ln,
                    layer_norm_first=layer_norm_first)
        
    else:
        print("Not implement {}".format(model_name))
    return model
