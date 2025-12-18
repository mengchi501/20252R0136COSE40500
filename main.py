import torch
import torch.nn as nn

import numpy as np

import os
import sys
import time
import logging
from datetime import datetime
from prettytable import PrettyTable

from src.replay_utils import *

from src.parse_args import args
from src.model.GCN.RGCN import RGCN
from src.model.GCN.replayRGCN import ReplayRGCN
from src.model.GCN.R2GCN import R2GCN
from src.model.GCN.R3GCN import R3GCN
from src.model.GCN.LARGCN import LARGCN
from src.model.GCN.distillRGCN import DistillRGCN

from src.data_load.KnowledgeGraph import KnowledgeGraph
from src.model.model_process import TrainBatchProcessor, EvalBatchProcessor
from src.model.model_process_vae import TrainBatchProcessorVAE, EvalBatchProcessorVAE

torch.autograd.set_detect_anomaly(True)


class Hub():
    def __init__(self, args):
        self.args = args

        # args.inverse = True
        # self.args.use_process_epoch_test = False

        # file_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = '/NAS/seungwon/CKGE_dataset'
        args.data_path = os.path.join(data_path, args.dataset)
        self.set_logger()

        print('Start loading Knowledge Graph')
        self.kg = KnowledgeGraph(args)
        print('Knowledge Graph successfully loaded')

        self.model = self.create_model()
        print('Model created.')
        self.args.logger.info(self.model)
        
        if args.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        else:
            pass

        self.args.logger.info(self.args)

    def create_model(self):
        if self.args.cl.lower() == 'replay':
            model = ReplayRGCN(self.args, self.kg)
        elif self.args.cl.lower() == 'replay_ewc':
            model = R3GCN(self.args, self.kg)
        elif self.args.cl.lower() == 'local_aug':
            model = LARGCN(self.args, self.kg)
        elif self.args.cl.lower() == 'ewc':
            model = R2GCN(self.args, self.kg)
        elif self.args.cl.lower() == 'memory_replay':
            model = RGCN(self.args, self.kg)
            self.args.use_memory_replay = True
        elif self.args.cl.lower() == 'distill':
            model = DistillRGCN(self.args, self.kg)
        else:
            model = RGCN(self.args, self.kg)
        model.to(self.args.gpu)
        return model

    def set_logger(self):
        self.args.log_path = os.path.join(args.log_path, datetime.now().strftime('%Y%m%d/'))
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset + '-' + args.gcn + '-' + args.decoder

        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = f'{self.args.log_path}_{args.trial}.log'

        print(logging_file_name)

        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

    def train(self):
        self.args.logger.info(f'Start training for snapshot {self.args.snapshot}.')

        self.best_valid = 0.0
        self.stop_epoch = 0

        train_time = 0.0
        total_train_time = 0.0

        if self.args.cl.lower() == 'local_aug':
            train_processor = TrainBatchProcessorVAE(self.args, self.kg)
        else:
            train_processor = TrainBatchProcessor(self.args, self.kg)
        self.args.valid = True
        if self.args.cl.lower() == 'local_aug':
            valid_processor = EvalBatchProcessorVAE(self.args, self.kg)
        else:
            valid_processor = EvalBatchProcessor(self.args, self.kg)

        '''Distillation / Freeze old entities & relations for distillation'''
        if self.args.cl.lower() == 'distill':
            self.model.freeze()
        
        for epoch in range(self.args.epoch_num):
            self.args.epoch = epoch

            '''Distillation / Unfreeze old entities & relations for distillation'''
            if self.args.cl.lower() == 'distill' and epoch > self.args.freeze_epoch:
                self.model.unfreeze()

            '''Train Epoch'''
            loss, train_epoch_time = train_processor.process_epoch(self.model, self.optimizer)
            train_time += train_epoch_time
            
            '''Validation Epoch'''
            if epoch > 0 and epoch % args.evaluate_every == 0:
                valid_res = valid_processor.process_epoch(self.model)

                total_train_time += train_time
                train_time = 0.0

                if self.best_valid < valid_res['mrr']:
                    self.best_valid = valid_res['mrr']
                    self.stop_epoch = max(0, self.stop_epoch - 5)

                    self.args.logger.info('Snapshot:{}\tEpoch:{}\tTrain Loss:{}\tValid MRR:{}\tValid Hits@1:{}\tValid Hits@10:{}\tBest:{}'.format(self.args.snapshot, epoch, loss, valid_res['mrr'], valid_res['hits1'], valid_res['hits10'], self.best_valid))

                    # save best model 
                    self.args.logger.info('\tSaving best model...')
                    torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch}, os.path.join(args.save_path, args.decoder, str(self.args.snapshot), f'best_mrr_model_{args.trial}.pth'))
                else:
                    self.args.logger.info('Snapshot:{}\tEpoch:{}\tTrain Loss:{}\tValid MRR:{}\tValid Hits@1:{}\tValid Hits@10:{}\tBest:{}'.format(self.args.snapshot, epoch, loss, valid_res['mrr'], valid_res['hits1'], valid_res['hits10'], self.best_valid))

                    self.stop_epoch += 1
                    if self.stop_epoch >= self.args.patience:
                        self.args.logger.info('Early stopping!!')
                        break

            '''Distillation / EMA for Distillation of Teacher Model'''
            if self.args.cl.lower() == 'distill':
                self.model.ema()

        self.args.valid = False

        return total_train_time

                
    def test(self):
        self.args.logger.info(f'Start testing for snapshot {self.args.snapshot_test}.')

        if self.args.cl.lower() == 'local_aug':
            test_processor = EvalBatchProcessorVAE(self.args, self.kg)
        else:
            test_processor = EvalBatchProcessor(self.args, self.kg)
        test_res = test_processor.process_epoch(self.model)

        return test_res

    # 다음 task 위해서 model 준비 
    def next_snapshot_setting(self):
        if self.args.cl.lower() == 'replay' or self.args.cl.lower() == 'replay_ewc':
            self.kg.reduce()
        if self.args.use_memory_replay:
            self.kg.store_memory(self.args.snapshot)
        self.kg.load_data(self.args.snapshot + 1)
        self.model.switch_snapshot()

    def reset_model(self, model=False, optimizer=False):
        '''
        Reset the model or optimizer
        :param model: If True: reset the model
        :param optimizer: If True: reset the optimizer
        '''
        # if model:
        #     self.model, self.optimizer = self.create_model()
        if optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.lr))

    # 수정해야 되는 부분!!! 
    def continual_learning(self):
        print("Start Continual Learning")

        report_results = PrettyTable()
        report_results.field_names = ['Snapshot', 'Time', 'Whole_MRR', 'Hits@1', 'Hits@3', 'Hits@10'] 

        # test_results : 각 task를 학습한 뒤 이전부터 현재까지의 task에 대한 성능을 나타내는 prettytable을 담는 list 
        test_results = []
        # training_times : 각 task에 대한 학습 시간을 담는 list 
        training_times = []
        # BWT, FWT : BWT, FWT를 계산하기 위해 성능값을 담는 list 
        BWT, FWT = [], []
        # first_learning_res : 처음 학습한 task에 대한 성능을 담는 list 
        first_learning_res = []

        for ss_id in range(int(self.args.snapshot_num)):
            self.args.snapshot = ss_id
            self.args.snapshot_test = ss_id

            self.model.pre_snapshot()
            if self.args.cl.lower() == 'local_aug':
                self.model.train_vae()

            if self.args.cl.lower() == 'replay' or self.args.cl.lower() == 'replay_ewc':
                self.model.combine()

            # 첫 task가 아닌 경우 현재 task를 학습하기 전의 성능 계산 
            if ss_id > 0: 
                self.args.test_FWT = True
                res_before = self.test()
                self.args.logger.info(f"FWT {self.args.snapshot} : {res_before['mrr']}")
                FWT.append(res_before['mrr'])
            self.args.test_FWT = False

            # train 단계 
            if ss_id > 0: # 지워야됨 !!!!!
                training_time = self.train()
            else:
                self.args.valid = False # 지워야 됨 !!!!
                training_time = 0 # 지워야 됨 !!!!

            # best model 불러오기 
            best_checkpoint = os.path.join(args.save_path, args.decoder, str(ss_id), f'best_mrr_model_{args.trial}.pth')
            self.load_checkpoint(best_checkpoint)

            self.model.snapshot_post_processing()

            # test 단계 
            test_res = PrettyTable()
            test_res.field_names = ['Snapshot:'+str(ss_id), 'MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']

            reses = [] # 현재까지 학습한 모든 task의 test set에 대한 성능 
            for test_ss_id in range(ss_id + 1):
                self.args.snapshot_test = test_ss_id
                
                res = self.test()

                if test_ss_id == ss_id:
                    first_learning_res.append(res['mrr'])
                test_res.add_row([test_ss_id, res['mrr'], res['hits1'], res['hits3'], res['hits5'], res['hits10']])
                reses.append(res)
            # 마지막에 BWT 계산
            # BWT는 ((현재 모델의 이전 Task에 대한 성능) - (해당 Task를 처음 학습했을 때 성능))의 평균 
            if ss_id == self.args.snapshot_num - 1:
                for iid in range(self.args.snapshot_num - 1):
                    BWT.append(reses[iid]['mrr'] - first_learning_res[iid])

            # 현재 모델에 대해서 수행한 test set들의 성능 
            self.args.logger.info('\n{}'.format(test_res))
            test_results.append(test_res)

            # 현재 모델에 대해서 수행한 test set들을 통틀어서 성능 계산 
            whole_mrr, whole_hits1, whole_hits3, whole_hits10 = self.get_report_result(reses)
            report_results.add_row([ss_id, training_time, whole_mrr, whole_hits1, whole_hits3, whole_hits10])
            training_times.append(training_time)

            # 다음 task 위해서 model 준비 
            if self.args.snapshot < int(self.args.snapshot_num) - 1:
                self.next_snapshot_setting()
                self.reset_model(optimizer=True)
        
        self.args.logger.info('Final Result:\n{}'.format(test_results))
        self.args.logger.info('Report Result:\n{}'.format(report_results))
        self.args.logger.info('Sum_Training_Time:{}'.format(sum(training_times)))
        self.args.logger.info('Every_Training_Time:{}'.format(training_times))
        self.args.logger.info('Forward transfer: {}  Backward transfer: {}'.format(sum(FWT)/len(FWT), sum(BWT)/len(BWT)))


    def get_report_result(self, results):
        '''
        Get report results of the final model: mrr, hits@1, hits@3, hits@10
        :param results: Evaluation results dict: {mrr: hits@k}
        :return: mrr, hits@1, hits@3, hits@10
        '''
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.snapshots[idx].test))
        whole_mrr = sum([mrr * num_test[i] for i, mrr in enumerate(mrrs)]) / sum(num_test)
        whole_hits1 = sum([hits1 * num_test[i] for i, hits1 in enumerate(hits1s)]) / sum(num_test)
        whole_hits3 = sum([hits3 * num_test[i] for i, hits3 in enumerate(hits3s)]) / sum(num_test)
        whole_hits10 = sum([hits10 * num_test[i] for i, hits10 in enumerate(hits10s)]) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))

if __name__ == '__main__':
    hub = Hub(args)
    hub.continual_learning()