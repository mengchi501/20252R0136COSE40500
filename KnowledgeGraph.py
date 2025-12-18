import torch
from torch_geometric.utils import degree

import numpy as np
import pickle

from ..utils import *
from copy import deepcopy as dcopy



class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_entities, self.num_relations = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()

        self.snapshots = {i: SnapShot(self.args) for i in range(int(self.args.snapshot_num))}
        self.combined_snapshots = {i: CombinedSnapShot(self.args) for i in range(int(self.args.snapshot_num))}
        self.reduced_snapshots = {i: ReducedSnapShot(self.args) for i in range(int(self.args.snapshot_num))}
        self.load_data(0)

    def load_data(self, ss_id):
        '''
        load data from all snapshot file \\
        should be called at the start of every snapshot 
        '''
        hr2t_all = dict()
        train_all, valid_all, test_all = [], [], []

        self.new_entities = set()

        # 추가
        self.new_relations = set()

        '''load facts, 아직 id로 변환되지 않은 facts'''
        train_facts = load_fact(os.path.join(self.args.data_path, str(ss_id), 'train.txt'))
        test_facts = load_fact(os.path.join(self.args.data_path, str(ss_id), 'test.txt'))
        valid_facts = load_fact(os.path.join(self.args.data_path, str(ss_id), 'valid.txt'))

        '''extract entities & relations from facts'''
        self.expand_entity_relation(train_facts)
        self.expand_entity_relation(valid_facts)
        self.expand_entity_relation(test_facts)

        # print(f'{ss_id}: expand_entity_relation() Done')

        '''read train/test/valid data, id로 변환'''
        train = self.fact2id(train_facts)
        valid = self.fact2id(valid_facts, order=True)
        test = self.fact2id(test_facts, order=True)

        # print(f'{ss_id}: fact2id() Done')

        '''
        Get edge_index and edge_type for GCN"
            edge_index = [[s_1, s_2, ... s_n],[o_1, o_2, ..., o_n]]
            edge_type = [r_1, r_2, ..., r_n]
        '''
        edge_h, edge_r, edge_t = [], [], []
        edge_h, edge_t, edge_r = self.expand_kg(train, 'train', edge_h, edge_t, edge_r, hr2t_all)
        edge_h, edge_t, edge_r = self.expand_kg(valid, 'valid', edge_h, edge_t, edge_r, hr2t_all)
        edge_h, edge_t, edge_r = self.expand_kg(test, 'test', edge_h, edge_t, edge_r, hr2t_all)

        # print(f'{ss_id}: expand_kg() Done')

        '''prepare data for 're-training' model'''
        train_all += train
        valid_all += valid
        test_all += test

        '''store this snapshot'''
        self.store_snapshot(ss_id, train, train_all, test, test_all, valid, valid_all, edge_h, edge_t, edge_r, hr2t_all)

        # print(f'{ss_id}: store_snapshot() Done')


        self.new_entities.clear()

        # 추가 
        self.new_relations.clear()

        # print(f'{ss_id} Done')

    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        for (s, r, o) in facts:
            '''extract entities'''
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_entities
                self.num_entities += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_entities
                self.num_entities += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_relations
                if self.args.inverse:
                    self.relation2id[r + '_inv'] = self.num_relations + 1
                    self.relation2inv[self.num_relations] = self.num_relations + 1
                    self.relation2inv[self.num_relations + 1] = self.num_relations
                    self.num_relations += 1
                self.num_relations += 1

    def fact2id(self, facts, order=False):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (s, r, o) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
                
                if self.args.inverse:
                    i = i + 2
                else:
                    i = i + 1
        else:
            for (s, r, o) in facts:
                fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    # def filter_golden_label(self, label):
    #     y = np.zeros([self.num_entities], dtype=np.float32)
    #     label = np.array(list(label))
    #     # print(label)
    #     y[label] = 1.0
    #     return torch.FloatTensor(y)

    def expand_kg(self, facts, split, edge_h, edge_t, edge_r, hr2t_all):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == 'train':
                '''edge_index'''
                edge_h.append(h)
                edge_r.append(r)
                edge_t.append(t)
            '''sr2o'''
            add_key2val(hr2t_all, (h, r), t)
            if self.args.inverse:
                add_key2val(hr2t_all, (t, self.relation2inv[r]), h)
        return edge_h, edge_t, edge_r

    def store_snapshot(self, ss_id, train_new, train_all, test, test_all, valid, valid_all, edge_h, edge_t, edge_r, hr2t_all):
        '''store snapshot data'''
        self.snapshots[ss_id].num_entities = dcopy(self.num_entities)
        self.snapshots[ss_id].num_relations = dcopy(self.num_relations)

        '''train, valid and test data'''
        self.snapshots[ss_id].train_new = np.array(train_new)
        self.snapshots[ss_id].train_all = np.array(train_all)
        self.snapshots[ss_id].test = np.array(test)
        self.snapshots[ss_id].valid = np.array(valid)
        self.snapshots[ss_id].valid_all = np.array(valid_all)
        self.snapshots[ss_id].test_all = np.array(test_all)

        '''임시!!!!! Master node와의 연결 추가하는 작업, 생략 가능!!!!'''
        

        '''edge_index, edge_type (for GCN of MEAN and LAN)'''
        self.snapshots[ss_id].edge_h = np.array(edge_h)
        self.snapshots[ss_id].edge_r = np.array(edge_r)
        self.snapshots[ss_id].edge_t = np.array(edge_t)

        '''sr2o (to filter golden facts)'''
        self.snapshots[ss_id].hr2t_all = dcopy(hr2t_all)
        self.snapshots[ss_id].edge_index = build_edge_index(edge_h, edge_t).to(self.args.gpu)
        self.snapshots[ss_id].edge_type = torch.cat(
            [torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(self.args.gpu)
        self.snapshots[ss_id].new_entities = dcopy(list(self.new_entities))

        self.snapshots[ss_id].new_relations = dcopy(list(self.new_relations))

        # importance calculation for cluster generation 
        if ss_id > 0:
            self.snapshots[ss_id].importance = dcopy(self.snapshots[ss_id - 1].importance)
        edge_index = self.snapshots[ss_id].edge_index.detach().cpu()
        deg = np.array(degree(edge_index[0], num_nodes=self.num_entities))
        for i in range(self.num_entities):
            self.snapshots[ss_id].importance[i] = self.snapshots[ss_id].importance.get(i, 0) + deg[i]

        self.snapshots[ss_id].match_edge_type = dict()
        src, dst = self.snapshots[ss_id].edge_index
        for i in range(len(src)):
            self.snapshots[ss_id].match_edge_type[(int(src[i]), int(dst[i]))] = self.snapshots[ss_id].edge_type[i]

        # Memory storing for replay
        if ss_id > 0 and self.args.use_memory_replay:
            with open(f"/NAS/seungwon/MyRGCN_CL/pickle_files/memory/trial{self.args.trial}_{ss_id - 1}.pkl", "rb") as f:
                memory = pickle.load(f)
            self.snapshots[ss_id].train_memory = np.vstack((self.snapshots[ss_id].train_new, memory))

            if self.args.use_for_conv:
                old_edge_h = list(memory[:, 0])
                old_edge_r = list(memory[:, 1])
                old_edge_t = list(memory[:, 2])
                old_edge_index = build_edge_index(old_edge_h, old_edge_t).to(self.args.gpu)
                old_edge_type = torch.cat(
                    [torch.LongTensor(old_edge_r), torch.LongTensor(old_edge_r) + 1]).to(self.args.gpu)
                self.snapshots[ss_id].edge_index = torch.cat((self.snapshots[ss_id].edge_index, old_edge_index), dim=1)
                self.snapshots[ss_id].edge_type = torch.cat((self.snapshots[ss_id].edge_type, old_edge_type), dim=0)
            

    def store_combined_snapshot(self, ss_id, num_entities, num_relations, edge_index, edge_type):
        self.combined_snapshots[ss_id].num_entities = dcopy(num_entities)
        self.combined_snapshots[ss_id].num_relations = dcopy(num_relations)

        self.combined_snapshots[ss_id].edge_index = dcopy(edge_index)
        self.combined_snapshots[ss_id].edge_type = dcopy(edge_type)

    def store_reduced_snapshot(self, ss_id, num_entities, num_relations, edge_index, edge_type):
        self.reduced_snapshots[ss_id].num_entities = dcopy(num_entities)
        self.reduced_snapshots[ss_id].num_relations = dcopy(num_relations)

        self.reduced_snapshots[ss_id].edge_index = dcopy(edge_index)
        self.reduced_snapshots[ss_id].edge_type = dcopy(edge_type)

    def store_memory(self, ss_id):
        if 'priority' in self.args.store_memory.lower():
            if self.args.store_memory == 'priority_high':
                important_entities = list(sorted(self.snapshots[ss_id].importance, key=self.snapshots[ss_id].importance.get, reverse=True))[:10]
            elif self.args.store_memory == 'priority_low':
                important_entities = list(sorted(self.snapshots[ss_id].importance, key=self.snapshots[ss_id].importance.get, reverse=False))[:1000]

            candidates = []
            for triple in self.snapshots[ss_id].train_new:
                h, r, t = triple
                if h in important_entities or t in important_entities:
                    candidates.append(triple)
            
            idx = np.random.choice(len(candidates), size=int(self.args.replay_size), replace=False)
            memory = np.array(candidates)[idx]
        elif self.args.store_memory.lower() == 'uniform':
            if ss_id > 0:
                with open(f"/NAS/seungwon/MyRGCN_CL/pickle_files/memory/trial{self.args.trial}_{ss_id - 1}.pkl", "rb") as f:
                    old_memory = pickle.load(f)
            memory = []

            size_per_snapshot = int(self.args.replay_size / (ss_id + 1))

            for i in range(ss_id):
                area = int(self.args.replay_size / ss_id)
                idx = np.random.choice(np.arange(area * i, area * (i + 1)), size=size_per_snapshot, replace=False)
                memory.append(old_memory[idx])
            idx = np.random.choice(len(self.snapshots[ss_id].train_new), size=size_per_snapshot, replace=False)
            memory.append(self.snapshots[ss_id].train_new[idx])
            memory = np.vstack(memory)
        else:
            idx = np.random.choice(len(self.snapshots[ss_id].train_new), size=int(self.args.replay_size), replace=False)
            memory = self.snapshots[ss_id].train_new[idx]
        with open(f"/NAS/seungwon/MyRGCN_CL/pickle_files/memory/trial{self.args.trial}_{ss_id}.pkl", "wb") as f:
            pickle.dump(memory, f)

class SnapShot():
    def __init__(self, args):
        self.args = args
        self.num_entities, self.num_relations = 0, 0
        self.train_new, self.train_all, self.test, self.valid, self.valid_all, self.test_all = list(), list(), list(), list(), list(), list()
        self.edge_h, self.edge_r, self.edge_t = [], [], []
        self.hr2t_all = dict()
        self.edge_index, self.edge_type = None, None
        self.new_entities = []
        self.new_relations = []

        self.importance = dict()
        self.train_memory = list()

class CombinedSnapShot():
    def __init__(self, args):
        self.args = args
        self.num_entities, num_relations = 0, 0
        self.edge_index, self.edge_type = None, None
        self.new_clusters = []
        self.new_relations = []

class ReducedSnapShot():
    def __init__(self, args):
        self.args = args
        self.num_entities, num_relations = 0, 0
        self.edge_index, self.edge_type = None, None
        self.new_clusters = []
        self.new_relations = []