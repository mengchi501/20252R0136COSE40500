import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from src.data_load.data_loader import TrainDataset, TestDataset

class EWC:
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        # initialize fisher matrix
        self.fisher = dict()
        self.param = dict()

    # def pre_snapshot(self):
    #     '''Consolidate parameter of prior task A'''
    #     if self.args.snapshot > 0:
    #         for n, p in self.named_parameters():
    #             n = n.replace('.','_')
    #             self.param[n] = p.data.detach().clone().to(self.args.gpu)
        
    #         self.args.snapshot -= 1
    #         self._compute_fisher()
    #         self.args.snapshot += 1

    # def _compute_fisher(self):
    #     dataset = TrainDataset(self.args, self.kg)
    #     data_loader = DataLoader(dataset,
    #                             shuffle=True,
    #                             # batch_size = self.args.batch_size,
    #                             batch_size=1,
    #                             collate_fn=dataset.collate_fn,
    #                             pin_memory=True)
    #     self.args.logger.info(f'batch size of DataLoader for FIM calc: {data_loader.batch_size}')

    #     for n, p in self.named_parameters():
    #         if p.requires_grad:
    #             n = n.replace('.', '_')
    #             self.fisher[n] = p.data.clone().detach().zero_().to(self.args.gpu) # set as same shape of parameter 
        
    #     self.eval()

    #     edge_index, edge_type = self.kg.snapshots[self.args.snapshot].edge_index, self.kg.snapshots[self.args.snapshot].edge_type # tensor(2, 2 * num_triples), tensor(2 * num_triples)

    #     for idx_b, batch in enumerate(data_loader):
    #         self.zero_grad()
    #         bh, br, bt, by = batch # batch = (batch_size, 4)
 
    #         x = self(edge_index, edge_type)
    
    #         bh = bh.to(self.args.gpu, non_blocking=True)
    #         br = br.to(self.args.gpu, non_blocking=True)
    #         bt = bt.to(self.args.gpu, non_blocking=True)
    #         by = by.to(self.args.gpu, non_blocking=True)
    #         batch_loss = self.loss(x,
    #                                bh,
    #                                br,
    #                                bt,
    #                                by if by is not None else by).float()

    #         batch_loss.backward()

    #         for n, p in self.named_parameters():
    #             if p.requires_grad:
    #                 n = n.replace('.', '_')
    #                 # F_i is the mean of (gradient)^2 across the dataset
    #                 self.fisher[n] += p.grad.data.clone().detach().pow(2)

    #     # Average the Fisher values over the number of samples
    #     num_data = len(data_loader.dataset)
    #     for n, f in self.fisher.items():
    #         self.fisher[n] = f / num_data

    #     # Restore model to training mode
    #     self.train()

    # def _compute_fisher_vae(self, x_ori):
    #     pairs = []
    #     for node in range(num_nodes):
    #         neighbors = torch.unique(torch.cat([src[dst==node], dst[src==node]]))
    #         for nbr in neighbors:
    #             pairs.append((node, nbr))

    #     pairs = torch.tensor(pairs) # shape (N_pairs, 2)
    #     dataset = TensorDataset(pairs)
    #     dataloader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True)

    #     self.eval()

    #     for idx_b, (batch, ) in enumerate(dataloader):
    #         self.zero_grad()
    #         nodes = batch[:, 0]
    #         neighs = batch[:, 1]
            
    #         x = x_ori[neighs]
    #         c = x_ori[nodes]

    #         if self.args.conditional:
    #             recon_x, means, log_vars, _ = self(x, c)
    #         else:
    #             recon_x, means, log_vars, _ = self(x)
            
    #         batch_loss = self.loss(recon_x, x, means, log_vars)

    #         batch_loss.backward()

    #         for n, p in self.named_parameters():
    #             if self.args.only_embedding and 'embedding' not in n:
    #                 continue
    #             if p.requires_grad:
    #                 n = n.replace('.', '_')
    #                 # F_i is the mean of (gradient)^2 across the dataset
    #                 self.fisher[n] += p.grad.data.clone().detach().pow(2)
        
    #     # Average the Fisher values over the number of samples
    #     num_data = len(dataloader.dataset)
    #     for n, f in self.fisher.items():
    #         self.fisher[n] = f / num_data

    #     self.train()

    def ewc_loss(self):
        '''
        Get regularization loss for all old paramters to constraint the update of old paramters.
        '''
        loss = 0
        for n, p in self.named_parameters():
            if self.args.only_embedding and 'embedding' not in n:
                continue
            if p.requires_grad:
                n = n.replace('.', '_')

                p_old = self.param[n]
                f = self.fisher[n]

                size = p_old.shape[0]
                loss += (f[:size] * (p[:size] - p_old).pow(2)).sum()                
        return loss

    # def loss(self, x, head, rel, tail, label):
    #     # Calculate base loss
    #     new_loss = self.loss(x, head, rel, tail, label)
        
    #     # Add EWC loss
    #     if self.args.snapshot > 0:
    #         ewc_loss = self.ewc_loss()
    #     else:
    #         ewc_loss = 0.0
            
    #     return new_loss + float(self.args.regular_weight) * ewc_loss