import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, RGCNConv, FastRGCNConv, GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import glorot, zeros

from src.parse_args import args
from src.model.decoder import DistMult, TransE
from src.model.BaseModel import BaseModel

import math

torch.autograd.set_detect_anomaly(True)

class RGCN(BaseModel):
    def __init__(self, args, kg, ss_id=0):
        super(RGCN, self).__init__(args, kg)
        self.args = args
        self.kg = kg

        num_entities = self.kg.snapshots[ss_id].num_entities
        num_relations_conv = self.kg.snapshots[ss_id].num_relations
        num_relations_emb = self.kg.snapshots[ss_id].num_relations

        if not self.args.inverse:
            num_relations_conv *= 2
            

        self.entity_embedding = nn.Embedding(num_entities, args.emb_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations_emb, args.emb_dim))
        
        # nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        print(f'num_relations for RGCNConv: {num_relations_conv}')

        self.init_layers(args.emb_dim, args.hid_dim, num_relations_conv)

        self.init_decoder()

    def init_layers(self, emb_dim, hid_dim, num_relations_conv):
        if args.gcn.lower() == 'fastrgcn':
            self.conv1 = FastRGCNConv(in_channels=emb_dim, 
                                out_channels=hid_dim, 
                                num_relations=num_relations_conv,
                                num_blocks=args.block_dim)
            self.conv2 = FastRGCNConv(in_channels=hid_dim, 
                                out_channels=emb_dim, 
                                num_relations=num_relations_conv,
                                num_blocks=args.block_dim)
        elif args.gcn.lower() == 'rgcn':
            self.conv1 = RGCNConv(in_channels=emb_dim, 
                                out_channels=hid_dim, 
                                num_relations=num_relations_conv,
                                num_blocks=args.block_dim)
            self.conv2 = RGCNConv(in_channels=hid_dim, 
                                out_channels=emb_dim, 
                                num_relations=num_relations_conv,
                                num_blocks=args.block_dim)
        elif args.gcn.lower() == 'gcn':
            self.conv1 = GCNConv(in_channels=emb_dim, 
                                out_channels=hid_dim, 
                                add_self_loops=True
                                )
            self.conv2 = GCNConv(in_channels=hid_dim, 
                                out_channels=emb_dim, 
                                add_self_loops=True
                                )
        
        if args.activ_func == 'relu':
            self.activ_func = F.relu
        else:
            self.activ_func = F.relu

    def init_decoder(self):
        if args.decoder.lower() == 'distmult':
            self.args.logger.info(f'\tuse DistMult as decoder.')
            self.decoder = DistMult()
        elif args.decoder.lower() == 'transe':
            self.args.logger.info(f'\tuse TransE as decoder.')
            self.decoder = TransE()
        else:
            self.decoder = DistMult()

    def forward(self, edge_index, edge_type):
        '''
        :param entity: entity indices
        :param edge_index: information of edges, contains 2 lists with the index of the source node & destination node respectively, (2, num_edges) 
        :param edge_type: the type of relations each edges have, (num_edges)
        :param edge_norm: normalizing constants of each edges, (num_edges)

        :returns score:
        '''

        x = self.entity_embedding.weight
        if args.gcn != 'gcn':
            x = self.conv1(edge_index=edge_index,
                        edge_type=edge_type, 
                        x=x)
        else:
            x = self.conv1(edge_index=edge_index, 
                        x=x)
        x = self.activ_func(x)
        x = F.dropout(x, p=args.dropout_general, training=self.training)
        if args.gcn != 'gcn':
            x = self.conv2(edge_index=edge_index,
                        edge_type=edge_type, 
                        x=x)
        else:
            x = self.conv2(edge_index=edge_index, 
                        x=x)
        
        return x

    def reg_loss(self, x):
        '''
        :param x: processed entity embeddings 
        '''
        return torch.mean(x.pow(2)) + torch.mean(self.relation_embedding.pow(2))
    
    def loss(self, x, head, rel, tail, label):
        loss = self.decoder.loss(x, self.relation_embedding, head, rel, tail, label) + args.penalty * self.reg_loss(x)
    
        return loss

    def predict(self, x, head, rel):
        return self.decoder.predict(x, self.relation_embedding, head, rel, self.kg)


    #=========== Continual Learning ===========#
    # Done 
    # Weight Regularization 할 거면 해당 내용 추가해야 됨 
    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                # self.register_buffer(
                #     f'old_data_{name}', value.clone().detach()
                # )
                self.register_buffer('old_data_{}_{}'.format(self.args.snapshot, name), value.clone().detach())

    # Done 
    def switch_snapshot(self):
        self.store_old_parameters()

        new_num_entities_emb = self.kg.snapshots[self.args.snapshot + 1].num_entities
        new_num_relations_emb = self.kg.snapshots[self.args.snapshot + 1].num_relations
        new_num_relations_conv = self.kg.snapshots[self.args.snapshot + 1].num_relations

        num_entities_emb = self.kg.snapshots[self.args.snapshot].num_entities
        num_relations_emb = self.kg.snapshots[self.args.snapshot].num_relations
        num_relations_conv = self.kg.snapshots[self.args.snapshot].num_relations
        new_num_relations_conv = self.kg.snapshots[self.args.snapshot + 1].num_relations

        if not self.args.inverse:
            num_relations_conv *= 2
            new_num_relations_conv *= 2

        new_ent_embeddings, new_rel_embeddings, new_conv1_weight, new_conv2_weight = self.expand_param_size(new_num_entities_emb, new_num_relations_emb, new_num_relations_conv)

        # Embedding 수정 
        new_ent_embeddings[:num_entities_emb] = torch.nn.Parameter(self.entity_embedding.weight.data).to(self.args.gpu)
        new_rel_embeddings[:num_relations_emb] = torch.nn.Parameter(self.relation_embedding.data).to(self.args.gpu)

        self.entity_embedding.weight = torch.nn.Parameter(new_ent_embeddings).to(self.args.gpu)
        self.relation_embedding = torch.nn.Parameter(new_rel_embeddings).to(self.args.gpu)

        # RGCNConvLayer 수정 
        new_conv1_weight[:num_relations_conv, :, :, :] = self.conv1.weight.data
        new_conv2_weight[:num_relations_conv, :, :, :] = self.conv2.weight.data

        # self.conv1.weight = torch.nn.Parameter(new_conv1_weight).to(self.args.gpu)
        # self.conv2.weight = torch.nn.Parameter(new_conv2_weight).to(self.args.gpu)
        self.conv1.weight = torch.nn.Parameter(new_conv1_weight)
        self.conv2.weight = torch.nn.Parameter(new_conv2_weight)

        self.conv1.num_relations = new_num_relations_conv
        self.conv2.num_relations = new_num_relations_conv

    def expand_param_size(self, new_ent_num, new_rel_emb_num, new_rel_conv_num):
        '''
        Expand the parameters of the model including entity_embedding, relation_embedding, RGCNConv weight
        '''
        # 증가시켜야 되는 거 
        # self.entity_embedding : (num_entities, emb_dim) -> (증가된 num_entities, emb_dim)
        # self.relation_embedding : (num_relations, emb_dim) -> (증가된 num_relations, emb_dim)
        # RGCNConv weight : (num_relations, block_dim, block_emb_dim, block_emb_dim) -> (증가된 num_relations * 2, block_dim, block_emb_dim, block_emb_dim)

        block_emb_dim = int(self.args.emb_dim / self.args.block_dim)
        block_hid_dim = int(self.args.hid_dim / self.args.block_dim)

        entity_embedding = nn.Embedding(new_ent_num, self.args.emb_dim).to(self.args.gpu)
        relation_embedding = nn.Parameter(torch.Tensor(new_rel_emb_num, self.args.emb_dim)).to(self.args.gpu)
        nn.init.xavier_uniform_(relation_embedding, gain=nn.init.calculate_gain('relu'))

        conv1_weight = torch.nn.Parameter(torch.Tensor(new_rel_conv_num, self.args.block_dim, block_emb_dim, block_hid_dim)).to(self.args.gpu)
        conv2_weight = torch.nn.Parameter(torch.Tensor(new_rel_conv_num, self.args.block_dim, block_hid_dim, block_emb_dim)).to(self.args.gpu)

        # torch_geometric.nn.conv.RGCN 에 정의된 초기화 방식에 따라 glorot으로 initialize 
        glorot(conv1_weight)
        glorot(conv2_weight)
        
        return entity_embedding.weight.data, relation_embedding.data, conv1_weight.data, conv2_weight.data
    
    def embedding(self, stage=None):
        return self.entity_embedding.weight, self.relation_embedding