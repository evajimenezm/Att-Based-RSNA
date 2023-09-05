import os
import torch
import wandb
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = '\"platform\"' 
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'false' 

#os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import matplotlib.pyplot as plt
import numpy as np
import random

from Execution_options import set_opts

from Load_datasets import RSNA_ICH_MILDataset, MNIST_MILDataset
from Setup_model import AttCNNMILModule, CNNMIL, AttentionModule, MeanModule, MaxModule
from Plot_utils import plot_bag_MNIST, plot_bag_RSNA, plot_att_MNIST, plot_att_RSNA, plot_att_dist

from Engine import Trainer, evaluate

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def build_model(args):

    cnn = CNNMIL(input_size=args.input_shape, cnn_name=args.cnn_name, network=args.network_type)
    if args.method == "AttentionBased":
        att = AttentionModule(D_dim = cnn.output_size, L_dim=args.L_dim)
    elif args.method == "Max":
        att = MaxModule(D_dim = cnn.output_size, L_dim=args.L_dim)
    elif args.method == "Mean":
        att = MeanModule(D_dim = cnn.output_size, L_dim=args.L_dim, device=args.device)
    else:
        ValueError('Unknown method')
    fc = torch.nn.Linear(cnn.output_size, 1)
    return AttCNNMILModule( cnn=cnn, att=att, fc=fc)

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(args):

    if args.dataset_name == 'mnist':
        train_dataset = MNIST_MILDataset(name=args.dataset_name, mode='train', num_bags=args.num_train_bags, bag_size=args.bag_size, obj_labels=args.obj_labels)
        val_dataset = MNIST_MILDataset(name=args.dataset_name, mode='train', num_bags=args.num_val_bags, bag_size=args.bag_size, obj_labels=args.obj_labels)
        test_dataset = MNIST_MILDataset(name=args.dataset_name, mode='test', num_bags=args.num_test_bags, bag_size=args.bag_size, obj_labels=args.obj_labels)
    elif args.dataset_name == 'rsna':
        train_dataset = RSNA_ICH_MILDataset(mode='train', validation=False, bag_size=args.bag_size)
        val_dataset = RSNA_ICH_MILDataset(mode='train', validation=True, bag_size=args.bag_size)
        test_dataset = RSNA_ICH_MILDataset(mode='test', bag_size=args.bag_size)
    else:
        raise ValueError('Unknown dataset name')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    if args.dataset_name == 'mnist':
        X, T_label, y_labels = train_dataset[1]
        plot_bag_MNIST(X, y_labels, figsize=(10, 10))
    elif args.dataset_name == 'rsna':
        X, T_label, y_labels = train_dataset[1]
        plot_bag_RSNA(X, T_label, y_labels, figsize=(10, 10))
    else:
        raise ValueError('Unknown dataset name')

    model = build_model(args)
    print(model)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.balance_loss:
        class_counts = train_dataset.dataset.get_class_counts()
        pos_weight = torch.FloatTensor([class_counts[0]/class_counts[1]])
        print('Using pos_weight=', pos_weight)
    else:
        pos_weight = None
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, criterion, optimizer, device=args.device, early_stop_patience=args.patience, lr_decay=args.lr_decay)
    trainer.train(args.epochs, train_dataloader, val_dataloader)

    best_model = trainer.get_best_model()

    if args.method == "AttentionBased":
        plot_att_dist(best_model, test_dataset, args.device, args.dataset_name)
        
        idx = 1
        X, T, y = train_dataset[idx]
        T_logits, S_pred = best_model(X.unsqueeze(dim=0).to(args.device))
        S_pred = S_pred.detach().cpu().numpy().astype(float).squeeze()
        S_pred = (S_pred - S_pred.min()) / (S_pred.max() - S_pred.min())
            
        if args.dataset_name == 'mnist':
            fig = plot_att_MNIST(S_pred, T, y.detach().cpu().numpy().astype(int).squeeze())
            plt.show()

            plot_bag_MNIST(X, y_labels, S_pred, figsize=(10, 10))
        elif args.dataset_name == 'rsna':
            fig = plot_att_RSNA(S_pred, T, y.detach().cpu().numpy().astype(int).squeeze())
            plt.show()

            plot_bag_RSNA(X, T_label, y_labels, S_pred, figsize=(10, 10))
        else:
            raise ValueError('Unknown dataset name')
    else:
        print('Attention: plot_att_dist and plot_att functions can not be used for this method!')


    if not os.path.exists(os.path.dirname(args.save_weights_path)):
        os.makedirs(os.path.dirname(args.save_weights_path))

    if torch.cuda.device_count() > 1:
        state_dict = best_model.module.state_dict()
    else:
        state_dict = best_model.state_dict()
    torch.save(state_dict, args.save_weights_path)

def test(args):

    if args.dataset_name == 'mnist':
        test_dataset = MNIST_MILDataset(name=args.dataset_name, mode='test', num_bags=args.num_test_bags, bag_size=args.bag_size, obj_labels=args.obj_labels)
    elif args.dataset_name == 'rsna':
        test_dataset = RSNA_ICH_MILDataset(mode='test', bag_size=args.bag_size)
    else:
        raise ValueError('Unknown dataset name')
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=None)

    model = build_model(args)
    print(model)
    weights_dict = torch.load(args.load_weights_path)
    model.load_state_dict(weights_dict)
    metrics = evaluate(model, test_dataloader, args.device)
    for metric in metrics:
        print('{:<25s}: {:s}'.format(metric, str(metrics[metric]))) 
    
    
def main():
    print('COMIENZA...')
    args = set_opts()

    print('Arguments:')
    for arg in vars(args):
        print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))
    
    args.save_weights_path = args.load_weights_path = args.weights_dir + f'/{args.dataset_name}/{args.cnn_name}/best.pt'

    seed_everything(args.set_seed)
    train(args)
    test(args)        
    
    print('Done!')

if __name__ == "__main__":
    main()
