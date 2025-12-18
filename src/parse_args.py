import argparse
import sys

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def decoder_name(d):
    if d.lower() == 'distmult':
        return 'DistMult'
    if d.lower() == 'transe':
        return 'TransE'


parser = argparse.ArgumentParser(allow_abbrev=False)

# Encoder 
parser.add_argument('--num_layers', default=2, type=int, help='number of convolution layers')
parser.add_argument('--emb_dim', default=100, type=int, help='embedding dimension of entity & relations')
parser.add_argument('--block_dim', default=5, type=int, help='dimension of blocks used in block decomposition')
parser.add_argument('--activ_func', default='relu', type=str, help='activation function')
parser.add_argument('--gcn', default='fastrgcn', type=str, help='type of GCN to use')

# GCN
parser.add_argument('--hid_dim', default=100, type=int, help='embedding dimension of hidden representation')

# Encoder regularization
parser.add_argument('--dropout_general', default=0.4, type=float, help='dropout rate for general edges')
parser.add_argument('--dropout_selfloop', default=0.2, type=float, help='dropout rate for self-loop edges')

# Decoder regularization 
parser.add_argument('--decoder', default='distmult', type=decoder_name, help='decoder name')
parser.add_argument('--penalty', default=0.01, type=float, help='penalty for l2')

# TransE
parser.add_argument('--margin', default=8.0, help='the margin of MarginLoss used for TransE')

# Optimizer 
parser.add_argument('--optimizer_name', default='Adam', help='optimizer to use')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

# Training 
parser.add_argument('--neg_ratio', default=1, type=int, help='how many negative samples to create per positive sample')
parser.add_argument('--filtered', default=True, type=bool, help='generate negative samples in filtered setting or not')
parser.add_argument('--epoch_num', default=10000, type=int, help='maximum number of epochs for training')
parser.add_argument('--evaluate_every', default=10, type=int, help='check validation for every specified epochs')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')

# Graph 
parser.add_argument('--bidirectional', default=True, type=str2bool, help='add inverse edges in order to make the graph bidirectional')
parser.add_argument('--split_ratio', default=0.5, type=float, help='ratio of sampling train data graph')
parser.add_argument('--inverse', default=True, type=str2bool, help='add inverse edges')

# Continual Learning
parser.add_argument('--snapshot_num', default=5, type=int, help='number of snapshots')
parser.add_argument('--cl', default="", type=str, help='which Continual learning method to use', choices=['replay', 'replay_ewc', 'local_aug', 'ewc', 'memory_replay', 'distill'])

# Continual Learning (GMR)
parser.add_argument('--simple', default=True, type=str2bool, help="whether to use the simple version of Graph Memory Replay")
parser.add_argument('--only_provide', default=True, type=str2bool, help="whether the cluster has bidirectional connection with the node during convolution")
parser.add_argument('--cluster_num', default=100, type=int, help='number of clusters to be created during reduce')

# Continual Learning (EWC)
parser.add_argument('--regular_weight', default=10, type=float, help='Regularization strength: alpha')
parser.add_argument('--only_embedding', default=False, type=str2bool, help='whether to use EWC only for embedding')

# Continual Learning (Replay)
parser.add_argument('--use_memory_replay', default=False, type=str2bool, help='whether to use replay')
parser.add_argument('--replay_size', default=1000, type=int, help='number of samples to be replayed')
parser.add_argument('--store_memory', default="random", type=str, help='how to store memory samples', choices=['random', 'priority_high', 'priority_low', 'uniform'])
parser.add_argument('--use_for_conv', default=False, type=str2bool, help='whether to use memory samples for convolution')

# Continual Learning (Distillation)
parser.add_argument('--distill_weight', default=0.1, type=float, help='weight for distillation loss')
parser.add_argument('--old_param_grad_scale', default=0.0, type=float, help='Scale of the gradient from original loss for old parameters (0.0=No Original Loss on Old Params)')
parser.add_argument('--use_freeze_epoch', default=False, type=str2bool, help='whether to use freeze epoch for distillation')
parser.add_argument('--freeze_epoch', default=200, type=int, help='number of epochs to freeze the embeddings of old entities & relations')
parser.add_argument('--use_ema', default=False, type=str2bool, help='whether to use exponential moving average for distillation')
parser.add_argument('--ema', default=0.999, type=float, help='exponential moving average for distillation')
parser.add_argument('--ema_start', default=200, type=int, help='number of epochs to start exponential moving average for distillation')
parser.add_argument('--distill_loss_type', default='mse', type=str, help='distillation loss type', choices=['mse', 'logit', 'cos', 'geometry'])
parser.add_argument('--use_mask_distill', default=False, type=str2bool, help='whether to use mask distillation loss for distillation')

# Augmentation (VAE)
parser.add_argument('--pretrain_lr', default=1e-5, type=float, help='learning rate for pretraining CVAE')
parser.add_argument('--pretrain_batch_size', default=8192, type=int, help='batch size for pretraining CVAE')
parser.add_argument('--conditional', default=True, type=str2bool, help='whether to use CVAE or VAE')
parser.add_argument('--latent_size', default=10, type=int, help='size of latent representation of VAE')
parser.add_argument('--total_iterations', default=10000, type=int, help='size of latent representation of VAE')
parser.add_argument('--sample_num', default=2, type=int, help='number of samples to be generated during training')

# General 
parser.add_argument('--gpu', default=0, type=int, help='select which gpu to use')
parser.add_argument('--dataset', default='ENTITY', type=str, help='dataset name')
parser.add_argument('--log_path', default='/NAS/seungwon/MyRGCN_CL/output/logs', type=str, help='path of directory where the logs will be saved at')
parser.add_argument('--save_path', default='/NAS/seungwon/MyRGCN_CL/output/best_models', type=str, help='path of directory where best models will be saved at')
parser.add_argument('--trial', default=0, type=int, help='trial number to identify model')
parser.add_argument('--use_process_epoch_test', default=False, type=str2bool, help='use use_process_epoch_test() for model_process')

args, _ = parser.parse_known_args()