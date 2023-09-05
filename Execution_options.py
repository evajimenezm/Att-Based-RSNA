import argparse
import torch

def custom_list(string):
    return [int(x) for x in string.split(',')]

def set_opts():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--num_workers', default=16, type=int, help="Number of workers to load data")

    # path settings
    # parser.add_argument('--weights_dir', default='/home/fran/work_fran/TFM_Eva/weights/', type=str, metavar='PATH', help="Path to save the model weights")   
    parser.add_argument('--weights_dir', default='./weights/', type=str, metavar='PATH', help="Path to save the model weights")   
    # parser.add_argument('--results_dir', default='/home/fran/work_fran/TFM_Eva/results/', type=str, metavar='PATH', help="Path to save the results") 
    parser.add_argument('--results_dir', default='./results/', type=str, metavar='PATH', help="Path to save the results") 

    # dataset settings
    parser.add_argument('--dataset_name', default='mnist', type=str, help="Dataset to use")
    parser.add_argument('--bag_size', type=int, default=10, help="Number of instances per bag")
    parser.add_argument('--num_train_bags', type=int, default=2000, help="Number of training bags")
    parser.add_argument('--num_val_bags', type=int, default=500, help="Number of validation bags")
    parser.add_argument('--num_test_bags', type=int, default=500, help="Number of testing bags")
    parser.add_argument('--obj_labels', type=custom_list, default=[0], help="Objective labels (labels to be designated as positive)")

    # model settings
    parser.add_argument('--cnn_name', type=str, default='cnn_3_0.0', help="Name of CNN")
    parser.add_argument('--L_dim', type=int, default=50, help="Value of L (hidden representation dimension for attention)")
    parser.add_argument('--method', type=str, default='AttentionBased', help="Method to be applied")
    parser.add_argument('--network_type', type=str, default='Simple', help="Network to be applied")
    
    # trainning settings
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size of training")
    parser.add_argument('--epochs', type=int, default=40, help="Training epochs")
    parser.add_argument('--balance_loss', action='store_true', help="Balance the loss using class weights")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="Decaying rate for the learning rate")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--set_seed', type=int, default=42, help="Set seed for everything")

    args = parser.parse_args()

    if args.dataset_name == 'mnist':
        args.input_shape = (1, 28, 28)
    elif args.dataset_name == 'rsna':
        args.input_shape = (3, 512, 512)
    else:
        raise ValueError('Unknown dataset name')

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print('Using device:', args.device)  

    return args