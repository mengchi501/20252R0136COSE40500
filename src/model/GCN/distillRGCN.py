import torch
import pickle
import torch.nn.functional as F
from src.model.GCN.RGCN import RGCN

import copy

class DistillRGCN(RGCN):
    def __init__(self, args, kg, ss_id=0):
        super(DistillRGCN, self).__init__(args, kg, ss_id)

        self.hook_handles = []
        self.masked_entities = None

    def pre_snapshot(self):
        if self.args.snapshot > 0:
            self.masked_entities = self.get_masked_entities()

    def forward_with_embedding(self, entity_embedding, edge_index, edge_type):
        x = entity_embedding
        if self.args.gcn != 'gcn':
            x = self.conv1(edge_index=edge_index,
                        edge_type=edge_type, 
                        x=x)
        else:
            x = self.conv1(edge_index=edge_index, 
                        x=x)
        x = self.activ_func(x)
        x = F.dropout(x, p=self.args.dropout_general, training=self.training)
        if self.args.gcn != 'gcn':
            x = self.conv2(edge_index=edge_index,
                        edge_type=edge_type, 
                        x=x)
        else:
            x = self.conv2(edge_index=edge_index, 
                        x=x)
        
        return x

    def distill_loss(self, x):
        x_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_x').to(self.args.gpu)
        rel_emb_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_rel_emb').to(self.args.gpu)

        distill_loss = F.mse_loss(x[:self.kg.snapshots[self.args.snapshot - 1].num_entities], x_old) + F.mse_loss(self.relation_embedding[:self.kg.snapshots[self.args.snapshot - 1].num_relations], rel_emb_old)
        return distill_loss

    def logit_distill_loss(self, x):
        x_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_x').to(self.args.gpu)
        rel_emb_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_rel_emb').to(self.args.gpu)

        with open(f'/NAS/seungwon/MyRGCN_CL/pickle_files/memory/trial20_{self.args.snapshot - 1}.pkl', 'rb') as f:
            sample_triples = torch.tensor(pickle.load(f)).to(self.args.gpu)
        
        head = sample_triples[:, 0]
        rel = sample_triples[:, 1]
        tail = sample_triples[:, 2]

        num_old_ent = self.kg.snapshots[self.args.snapshot - 1].num_entities

        teacher_scores = self.decoder.predict(x_old, rel_emb_old, head, rel, self.kg) # (num_samples, num_entities)
        student_scores = self.decoder.predict(x[:num_old_ent], self.relation_embedding, head, rel, self.kg) # (num_samples, num_entities)

        # Calculate KL Divergence Loss
        # Input: Log probabilities of student (Q)
        # Target: Probabilities of teacher (P)
        log_student_probs = F.log_softmax(student_scores, dim=1)
        teacher_probs = F.softmax(teacher_scores, dim=1)

        # KL(P || Q) = sum(P * (log P - log Q)) = sum(P * log P) - sum(P * log Q)
        # nn.KLDivLoss expects input as log_probs and target as probs (if log_target=False).
        # reduction='batchmean' divides by batch size, which is mathematically correct for KL divergence averaged over samples.
        loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean')
        
        return loss

    def cos_distill_loss(self, x):
        x_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_x').to(self.args.gpu)
        rel_emb_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_rel_emb').to(self.args.gpu)

        distill_loss = F.cosine_embedding_loss(x[:self.kg.snapshots[self.args.snapshot - 1].num_entities], x_old, torch.ones(x_old.shape[0]).to(self.args.gpu)) + F.cosine_embedding_loss(self.relation_embedding[:self.kg.snapshots[self.args.snapshot - 1].num_relations], rel_emb_old, torch.ones(rel_emb_old.shape[0]).to(self.args.gpu))
        return distill_loss

    def ent_distill_loss(self, x):
        x_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_x').to(self.args.gpu)

        num_old_ent = self.kg.snapshots[self.args.snapshot - 1].num_entities

        cos_loss = F.cosine_embedding_loss(x[:num_old_ent], x_old, torch.ones(x_old.shape[0]).to(self.args.gpu))
        # mag_loss = F.l1_loss(torch.norm(x_old, p=2, dim=1), torch.norm(x[:num_old_ent], p=2, dim=1))

        # return cos_loss + 0.2 * mag_loss
        return cos_loss
    
    def rel_distill_loss(self):
        rel_emb_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_rel_emb').to(self.args.gpu)

        num_old_rel = self.kg.snapshots[self.args.snapshot - 1].num_relations

        rel_loss = F.mse_loss(self.relation_embedding[:num_old_rel], rel_emb_old)
        return rel_loss

    def geometry_distill_loss(self, x):
        return self.ent_distill_loss(x) + self.rel_distill_loss()

    def loss(self, x, head, rel, tail, label):
        if self.args.snapshot > 0:
            scale = self.args.old_param_grad_scale
            num_old_ent = self.kg.snapshots[self.args.snapshot - 1].num_entities
            num_old_rel = self.kg.snapshots[self.args.snapshot - 1].num_relations
            
            # 1. Scale gradients for Entity Representations (x)
            # x is (num_entities, emb_dim)
            x_old = x[:num_old_ent]
            x_new = x[num_old_ent:]
            
            # Gradients back to old params will be scaled by 'scale'
            x_old_scaled = x_old * scale + x_old.detach() * (1 - scale)
            x_in = torch.cat([x_old_scaled, x_new], dim=0)
            
            # 2. Scale gradients for Relation Embeddings
            r = self.relation_embedding
            r_old = r[:num_old_rel]
            r_new = r[num_old_rel:]
            
            r_old_scaled = r_old * scale + r_old.detach() * (1 - scale)
            r_in = torch.cat([r_old_scaled, r_new], dim=0)
            
            # 4. Compute Original Loss using scaled inputs
            original_r_param = self.relation_embedding
            # self.relation_embedding is a Parameter. To swap it with a Tensor for calculation, 
            # we must delete it from the object first to avoid TypeError.
            del self.relation_embedding 
            self.relation_embedding = r_in
            
            try:
                ori_loss = super().loss(x_in, head, rel, tail, label)
            finally:
                # Restore
                if hasattr(self, 'relation_embedding'):
                    del self.relation_embedding
                self.relation_embedding = original_r_param
            
            # 4. Compute Distill Loss (affects old params)
            if self.args.use_mask_distill and self.masked_entities is not None:
                # To prevent distillation loss from affecting masked_entities, we construct a 
                # 'masked' embedding tensor where gradients are blocked for masked_entities.
                # However, we must use the original values for the forward pass, just detached from graph.
                
                emb = self.entity_embedding.weight
                mask = torch.ones(emb.shape[0], 1, device=self.args.gpu)
                mask[self.masked_entities] = 0
                
                # emb_prime has gradients enabled for unmasked, but disabled (detached) for masked.
                # emb * mask: keeps grads for unmasked (mask=1), zeros for masked (mask=0)
                # emb.detach() * (1 - mask): adds back the values for masked, but without grad history.
                emb_prime = emb * mask + emb.detach() * (1 - mask)
                
                edge_index = self.kg.snapshots[self.args.snapshot].edge_index
                edge_type = self.kg.snapshots[self.args.snapshot].edge_type
                
                # Re-compute x for distillation using the gradient-masked embedding
                x_for_distill = self.forward_with_embedding(emb_prime, edge_index, edge_type)
            else:
                x_for_distill = x

            if self.args.distill_loss_type == 'logit':
                dis_loss = self.args.distill_weight * self.logit_distill_loss(x_for_distill)
            elif self.args.distill_loss_type == 'cos':
                dis_loss = self.args.distill_weight * self.cos_distill_loss(x_for_distill)
            elif self.args.distill_loss_type == 'geometry':
                dis_loss = self.args.distill_weight * self.geometry_distill_loss(x_for_distill)
            else:
                dis_loss = self.args.distill_weight * self.distill_loss(x_for_distill)
            
            if self.args.epoch % 10 == 0: 
                print(f"ori_loss: {ori_loss.item():.4f}, dis_loss: {dis_loss.item():.4f}")

            return ori_loss + dis_loss
        else:
            return super().loss(x, head, rel, tail, label)
    
    #=========== Continual Learning ===========#
    
    def store_old_parameters(self):
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot].edge_index, self.kg.snapshots[self.args.snapshot].edge_type

        x_old = self.forward(edge_index, edge_type)
        rel_emb_old = self.relation_embedding

        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                if name == 'entity_embedding_weight':
                    self.register_buffer(f'old_data_{self.args.snapshot}_x', x_old.data.clone().detach())
                    continue
                if name == 'relation_embedding':
                    self.register_buffer(f'old_data_{self.args.snapshot}_rel_emb', rel_emb_old.data.clone().detach())
                    continue
                value = param.data
                self.register_buffer('old_data_{}_{}'.format(self.args.snapshot, name), value.clone().detach())

    def get_masked_entities(self):
        old_ent = set(self.kg.snapshots[self.args.snapshot - 1].new_entities)
        new_ent = set(self.kg.snapshots[self.args.snapshot].new_entities).difference(set(self.kg.snapshots[self.args.snapshot-1].new_entities))
        untouched_new = copy.deepcopy(new_ent)
        for (h, r, t) in self.kg.snapshots[self.args.snapshot].train_new:
            if (h in new_ent and t in old_ent) or (t in new_ent and h in old_ent):
                if h in new_ent and h in untouched_new:
                    untouched_new.remove(h)
                if t in new_ent and t in untouched_new:
                    untouched_new.remove(t)
        untouched_new = torch.tensor(list(untouched_new)).to(self.args.gpu)
        
        return untouched_new

    def freeze(self):
        if self.args.use_freeze_epoch and self.args.snapshot > 0:
            if not self.hook_handles:
                prev_ent_num = self.kg.snapshots[self.args.snapshot - 1].num_entities
                prev_rel_num = self.kg.snapshots[self.args.snapshot - 1].num_relations
                
                def get_mask_hook(limit):
                    def hook(grad):
                        new_grad = grad.clone()
                        new_grad[:limit] = 0
                        return new_grad
                    return hook

                h1 = self.entity_embedding.weight.register_hook(get_mask_hook(prev_ent_num))
                h2 = self.relation_embedding.register_hook(get_mask_hook(prev_rel_num))
                # h3 = self.conv1.weight.register_hook(get_mask_hook(prev_rel_num))
                # h4 = self.conv2.weight.register_hook(get_mask_hook(prev_rel_num))
                self.hook_handles.extend([h1, h2])

                self.args.logger.info(f"Freeze old entities ({prev_ent_num}) and relations ({prev_rel_num})")

    def unfreeze(self):
        if self.hook_handles:
            for h in self.hook_handles: 
                h.remove()
            self.hook_handles = []
            self.args.logger.info('Unfreeze entity and relation embeddings')
            
    def ema(self):
        if self.args.use_ema and self.args.snapshot > 0 and self.args.epoch >= self.args.ema_start:
            if self.args.epoch == self.args.ema_start:
                self.args.logger.info("Start EMA for Teacher model")
            x_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_x').to(self.args.gpu)
            rel_emb_old = self.get_buffer(f'old_data_{self.args.snapshot - 1}_rel_emb').to(self.args.gpu)

            prev_ent_num = self.kg.snapshots[self.args.snapshot - 1].num_entities
            prev_rel_num = self.kg.snapshots[self.args.snapshot - 1].num_relations

            x_new = self.args.ema * x_old + (1 - self.args.ema) * self.entity_embedding.weight.data[:prev_ent_num]
            rel_emb_new = self.args.ema * rel_emb_old + (1 - self.args.ema) * self.relation_embedding.data[:prev_rel_num]

            x_old.copy_(x_new)
            rel_emb_old.copy_(rel_emb_new)