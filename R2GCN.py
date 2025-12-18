from src.model.CL.EWC import EWC
from src.model.GCN.RGCN import RGCN

from torch.utils.data import DataLoader
from src.data_load.data_loader import TrainDataset, TestDataset

class R2GCN(RGCN, EWC):
    '''
    Regularization + RGCN    
    '''
    def __init__(self, args, kg):
        RGCN.__init__(self, args, kg)
        EWC.__init__(self, args, kg)

    # def pre_snapshot(self):
    #     '''Consolidate parameter of prior task A'''
    #     if self.args.snapshot > 0:
    #         for n, p in self.named_parameters():
    #             if self.args.only_embedding and 'embedding' not in n:
    #                 continue
    #             n = n.replace('.','_')
    #             self.param[n] = p.data.detach().clone().to(self.args.gpu)
        
    #         self.args.snapshot -= 1
    #         self._compute_fisher()
    #         self.args.snapshot += 1

    def snapshot_post_processing(self):
        '''Consolidate parameter of current task A(to-be-prior)'''
        for n, p in self.named_parameters():
            if self.args.only_embedding and 'embedding' not in n:
                continue
            n = n.replace('.','_')
            self.param[n] = p.data.detach().clone().to(self.args.gpu)
    
        self._compute_fisher()

    def _compute_fisher(self):
        dataset = TrainDataset(self.args, self.kg)
        data_loader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=self.args.batch_size,
                                collate_fn=dataset.collate_fn,
                                pin_memory=True)
        self.args.logger.info(f'batch size of DataLoader for FIM calc: {data_loader.batch_size}')

        for n, p in self.named_parameters():
            if self.args.only_embedding and 'embedding' not in n:
                continue
            if p.requires_grad:
                n = n.replace('.', '_')
                self.fisher[n] = p.data.clone().detach().zero_().to(self.args.gpu) # set as same shape of parameter 
        
        self.eval()

        edge_index, edge_type = self.kg.snapshots[self.args.snapshot].edge_index, self.kg.snapshots[self.args.snapshot].edge_type # tensor(2, 2 * num_triples), tensor(2 * num_triples)

        for idx_b, batch in enumerate(data_loader):
            self.zero_grad()
            bh, br, bt, by = batch # batch = (batch_size, 4)
            current_batch_size = bh.shape[0]
 
            x = self(edge_index, edge_type)
    
            bh = bh.to(self.args.gpu, non_blocking=True)
            br = br.to(self.args.gpu, non_blocking=True)
            bt = bt.to(self.args.gpu, non_blocking=True)
            by = by.to(self.args.gpu, non_blocking=True)
            batch_loss = RGCN.loss(self, 
                                   x,
                                   bh,
                                   br,
                                   bt,
                                   by if by is not None else by).float()

            batch_loss.backward()

            for n, p in self.named_parameters():
                if self.args.only_embedding and 'embedding' not in n:
                    continue
                if p.requires_grad:
                    n = n.replace('.', '_')
                    # F_i is the mean of (gradient)^2 across the dataset
                    # We approximate F_i by multiplying the squared batch gradient by the batch size
                    self.fisher[n] += p.grad.data.clone().detach().pow(2) * current_batch_size

        # Average the Fisher values over the number of batches
        num_batches = len(data_loader)
        for n, f in self.fisher.items():
            self.fisher[n] = f / num_batches

        print(f'num_batches: {num_batches}')

        # Restore model to training mode
        self.train()

    def loss(self, x, head, rel, tail, label):
        # Calculate base loss
        new_loss = RGCN.loss(self, x, head, rel, tail, label)
        
        # Add EWC loss
        if self.args.snapshot > 0:
            ewc_loss = EWC.ewc_loss(self)
        else:
            ewc_loss = 0.0
            
        # print(f'new_loss: {new_loss}, ewc_loss: {float(self.args.regular_weight) * ewc_loss} total_loss: {new_loss + float(self.args.regular_weight) * ewc_loss}')

        return new_loss + float(self.args.regular_weight) * ewc_loss