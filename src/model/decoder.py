import torch
import torch.nn as nn
import torch.nn.functional as F

from src.parse_args import args

class Decoder:
    def __init__(self):
        pass

    def __call__(self, entity_embedding, relation_embedding, triplets):
        '''
        given the entity embeddings, relation embeddings and triplets, calculate the score of each triplet \\
        used during training to calculate loss 
        
        :param entity_embedding: the embeddings of entities 
        :param relation_embedding: the embeddings of relations 
        :param triplets: the triplets to calculate the scores of

        :returns score:
        '''
        pass

    def calc_score(self, ent_emb, rel_emb, entity_embedding):
        '''
        calculate the similarity between the query embeddings and the tail embeddings \\
        used during evaluation to rank candidate entities for each query 

        :param ent_emb: the embeddings of entities, (num_queries, emb_dim)
        :param rel_emb: the embeddings of relations, (num_queries, emb_dim)
        :param entity_embedding: the embedding of candidate entities, (num_entities, emb_dim)

        :returns res: the score of every candidate entities for each queries 
        '''
        pass

    def loss(self, ent_emb, rel_emb, head, rel, tail, label):
        pass

    def predict(self, ent_emb, rel_emb, head, rel, kg):
        pass

class DistMult(Decoder):
    def __init__(self):
        pass

    def loss(self, ent_emb, rel_emb, head, rel, tail, label):
        h_emb = ent_emb[head] # (num_triples, emb_dim)
        r_emb = rel_emb[rel]
        t_emb = ent_emb[tail]

        score = torch.sum(h_emb * r_emb * t_emb, dim=1) # (num_triples)
        # 추가됨
        # label = torch.as_tensor(label, dtype=float, device=f'cuda:{args.gpu}')

        return F.binary_cross_entropy_with_logits(score, label)
    
    def predict(self, ent_emb, rel_emb, head, rel, kg):
        if args.valid:
            num_entities = kg.snapshots[args.snapshot].num_entities
            # print(f'num_entities from predict(): {num_entities}')
        else:
            num_entities = kg.snapshots[args.snapshot_test].num_entities

        h_emb = ent_emb[head] # (num_queries, emb_dim) 
        r_emb = rel_emb[rel] # (num_queries, emb_dim) 
        cand_emb = ent_emb[:num_entities] # (num_entities, emb_dim) 

        query_emb = h_emb * r_emb # (num_queries, emb_dim) 
        query_emb = query_emb.unsqueeze(1) # (num_queries, 1, emb_dim)

        res = query_emb * cand_emb # (num_queries, num_entities, emb_dim)
        res = torch.sum(res, dim=-1).to(args.gpu) # (num_queries, num_entities)
        # print(f'shape of res: {res.shape}')

        return res
    
    def calc_score(self, ent_emb, rel_emb, cand_emb):
        query_emb = ent_emb * rel_emb
        query_emb = query_emb.unsqueeze(1) # (num_queries, 1, emb_dim)

        res = query_emb * cand_emb # (num_queries, num_entities, emb_dim)
        res = torch.sum(res, dim=-1).to(args.gpu) # (num_queries, num_entities)

        return res
    
class TransE(Decoder):
    def __init__(self):
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")

    def __call__(self, entity_embedding, relation_embedding, triplets):
        h,r,t = triplets.transpose()

        # device = entity_embedding.device
        # h_idx = torch.as_tensor(h, dtype=torch.long, device=device)
        # r_idx = torch.as_tensor(r, dtype=torch.long, device=device)
        # t_idx = torch.as_tensor(t, dtype=torch.long, device=device)
        h_idx = torch.as_tensor(h, dtype=torch.long).to(args.gpu)
        r_idx = torch.as_tensor(r, dtype=torch.long).to(args.gpu)
        t_idx = torch.as_tensor(t, dtype=torch.long).to(args.gpu)

        h_emb = entity_embedding[h_idx]  # (num_triplets, emb_dim)
        r_emb = relation_embedding[r_idx]  # (num_triplets, emb_dim)
        t_emb = entity_embedding[t_idx]  # (num_triplets, emb_dim)

        h_emb = self.norm_ent(h_emb)
        r_emb = self.norm_rel(r_emb)
        t_emb = self.norm_ent(t_emb)

        score = args.margin - torch.norm(h_emb + r_emb - t_emb, 1, -1)

        return score

    def split_pn_score(self, score, label):
        """
        split postive triples and negtive triples
        :param score: scores of all facts
        :param label: postive facts: 1, negtive facts: -1
        """
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label <= 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def loss(self, ent_emb, rel_emb, head, rel, tail, label):
        h_emb = ent_emb[head] # (num_triples, emb_dim)
        r_emb = rel_emb[rel]
        t_emb = ent_emb[tail]

        h_emb = self.norm_ent(h_emb)
        r_emb = self.norm_rel(r_emb)
        t_emb = self.norm_ent(t_emb)

        l = h_emb.shape[0]

        score = torch.norm(h_emb + r_emb - t_emb, 1, -1)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(args.gpu)

        return self.margin_loss_func(p_score, n_score, y) / l

    def predict(self, ent_emb, rel_emb, head, rel, kg):
        if args.valid:
            num_entities = kg.snapshots[args.snapshot].num_entities
            # print(f'num_entities from predict(): {num_entities}')
        else:
            num_entities = kg.snapshots[args.snapshot_test].num_entities

        h_emb = ent_emb[head] # (num_queries, emb_dim) 
        r_emb = rel_emb[rel] # (num_queries, emb_dim) 
        cand_emb = ent_emb[:num_entities] # (num_entities, emb_dim) 

        h_emb = self.norm_ent(h_emb)
        r_emb = self.norm_rel(r_emb)
        cand_emb = self.norm_ent(cand_emb)

        query_emb = h_emb + r_emb # (num_queries, emb_dim)
        query_emb = query_emb.unsqueeze(1) # (num_queries, 1, emb_dim) 

        res = query_emb - cand_emb # (num_queries, num_entities, emb_dim)
        res = args.margin - torch.norm(res, 1, -1) # (num_queries, num_entities) 

        return res


    def calc_score(self, ent_emb, rel_emb, cand_emb):
        ent_emb = self.norm_ent(ent_emb)
        rel_emb = self.norm_rel(rel_emb)
        cand_emb = self.norm_ent(cand_emb)

        query_emb = ent_emb + rel_emb # (num_queries, emb_dim)
        query_emb = query_emb.unsqueeze(1) # (num_queries, 1, emb_dim) 

        res = query_emb - cand_emb # (num_queries, num_entities, emb_dim)
        res = args.margin - torch.norm(res, 1, -1) # (num_queries, num_entities) 

        return res


    def norm_rel(self, r):
        return nn.functional.normalize(r, 2, -1)

    def norm_ent(self, e):
        return nn.functional.normalize(e, 2, -1)