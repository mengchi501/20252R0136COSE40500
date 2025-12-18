from torch.utils.data import Dataset
from src.utils import *

class TestDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg

        '''prepare data for validation and testing'''
        self.valid, self.test = self.build_facts()

    def __len__(self):
        if self.args.valid:
            return len(self.valid[self.args.snapshot])
        else:
            return len(self.test[self.args.snapshot_test])

    def __getitem__(self, idx):
        if self.args.valid:
            ele = self.valid[self.args.snapshot][idx]
        else:
            ele = self.test[self.args.snapshot_test][idx]
        fact, label = torch.LongTensor(ele['fact']), ele['label']
        label = self.get_label(label)

        return fact[0], fact[1], fact[2], label

    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        return h, r, t, label

    def get_label(self, label):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if self.args.valid:
            y = np.zeros([self.kg.snapshots[self.args.snapshot].num_entities], dtype=np.float32)
        else:
            y = np.zeros([self.kg.snapshots[self.args.snapshot_test].num_entities], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid, test = [], []
        for ss_id in range(int(self.args.snapshot_num)):
            valid_, test_ = list(), list()

            for (h, r, t) in self.kg.snapshots[ss_id].valid:
                valid_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            if self.args.inverse:
                for (h, r, t) in self.kg.snapshots[ss_id].valid:
                    valid_.append({'fact': (t, r+1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r+1)]})
            

            for (h, r, t) in self.kg.snapshots[ss_id].test:
                test_.append({'fact': (h, r, t), 'label': self.kg.snapshots[ss_id].hr2t_all[(h, r)]})
            if self.args.inverse:
                for (h, r, t) in self.kg.snapshots[ss_id].test:
                    test_.append({'fact': (t, r+1, h), 'label': self.kg.snapshots[ss_id].hr2t_all[(t, r+1)]})

            valid.append(valid_)
            test.append(test_)
        return valid, test
    

class TrainDataset(Dataset):
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        self.facts_new = self.build_facts()

    def __len__(self):
        return len(self.facts_new[self.args.snapshot])

    def __getitem__(self, idx):
        '''
        returns a single item(data sample)

        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        ele = self.facts_new[self.args.snapshot][idx]
        fact, label = ele['fact'], ele['label']

        '''negative sampling'''
        fact, label = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        # fact, label = torch.as_tensor(fact, dtype=int, device=f'cuda:{args.gpu}'), torch.as_tensor(label, dtype=float, device=f'cuda:{args.gpu}')
        return fact, label, None, None

    @staticmethod
    def collate_fn(batch):
        '''
        batch: (batch_size, 4)
        '''
        fact = torch.cat([_[0] for _ in batch], dim=0)
        label = torch.cat([_[1] for _ in batch], dim=0)
        return fact[:,0], fact[:,1], fact[:,2], label

    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        facts_new = list()
        for ss_id in range(int(self.args.snapshot_num)):
            facts_new_ = list()
            '''for LKGE and other baselines'''
            if ss_id > 0 and self.args.use_memory_replay:
                for h, r, t in self.kg.snapshots[ss_id].train_memory:
                    facts_new_.append({'fact':(h, r, t), 'label':1})
                    if self.args.inverse:
                        facts_new_.append({'fact': (t, r+1, h), 'label': 1})
            else:
                for h, r, t in self.kg.snapshots[ss_id].train_new:
                    facts_new_.append({'fact':(h, r, t), 'label':1})
                    if self.args.inverse:
                        facts_new_.append({'fact': (t, r+1, h), 'label': 1})
            facts_new.append(facts_new_)
        return facts_new

    def corrupt(self, fact):
        '''
        :param fact: positive facts
        :return: positive facts & negative facts ; pos/neg labels.
        '''
        ss_id = self.args.snapshot
        h, r, t = fact
        prob = 0.5

        '''random corrupt subject or object entities'''
        neg_h = np.random.randint(0, self.kg.snapshots[ss_id].num_entities - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.snapshots[ss_id].num_entities - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]

        '''get labels'''
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(0)
        return facts, label