from datetime import datetime
from json import decoder
import math
import os
import random
import sys
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse


from utility.parser import parse_args
from Models import *
from utility.batch_test import *
from utility.logging import Logger
args = parse_args()



class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.task_name = "%s_%s_%s" % (self.start_time, args.dataset, args.model_name)
        self.save_file_name = 'uni_results.csv'
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(data_config['norm_adj']).float().cuda()
        
        image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))
        text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))
        if args.model_name in ['EgoGCN', 'GRCN']:
            edges = np.load(f'../data/mmgcn/{args.dataset}/train.npy', allow_pickle=True)
            self.model = eval(args.model_name)(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, self.norm_adj, edge_index=edges)
        else:
            self.model = eval(args.model_name)(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, self.norm_adj)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.score_logger = []
        self.avg_score_logger = []
        self.avg_recall_logger = []


    def save_results(self, ret):
        with open(self.save_file_name, 'a') as f:
            f.writelines('%s,%s,%s,train=%d,%.4f,%.4f,%.4f\n' % (self.start_time, args.dataset, args.model_name, args.train_type, 
            ret['recall'][1], ret['ndcg'][1], ret['precision'][1]))

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model(training=args.train_type)
            result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
            return result

    def train(self):
        training_time_list = []
        stopping_step = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss, contrastive_loss = 0., 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1
                ua_embeddings, ia_embeddings = self.model(training=args.train_type)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss  

                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue


            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()


            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f" % (eval(args.Ks)[1], test_ret['recall'][1]))
                stopping_step = 0
                if args.save_model:
                    if not os.path.exists('../teacher_models'):
                        os.makedirs('../teacher_models')
                    torch.save({'state_dict': self.model.state_dict()} ,'../teacher_models/%s_%s_%d' % (args.model_name, args.dataset, args.train_type))
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break

        self.logger.logging(str(test_ret))
        self.save_results(test_ret)

    def bpr_loss(self, users, pos_items, neg_items):        
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)


        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    config['norm_adj'] = norm_adj
    config['mean_adj'] = mean_adj

    trainer = Trainer(data_config=config)
    trainer.train()

