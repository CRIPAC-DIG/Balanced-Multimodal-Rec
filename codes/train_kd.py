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


from utility.parser import parse_args
from Models import *
from utility.batch_test import *
from utility.logging import Logger
args = parse_args()


def KLDiverge(tpreds, spreds, distillTemp):
	tpreds = (tpreds / distillTemp).sigmoid()
	spreds = (spreds / distillTemp).sigmoid()
	return -(tpreds * (spreds + 1e-8).log() + (1 - tpreds) * (1 - spreds + 1e-8).log()).mean()

class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.task_name = "%s_%s_%s" % (self.start_time, args.dataset, args.model_name)
        self.save_file_name = 'kd_results.csv'
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
        else:
            edges = None

        self.model = eval(args.model_name)(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, self.norm_adj, edge_index=edges)

        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


        self.textual_teacher = eval(args.model_name)(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, self.norm_adj, edge_index=edges)
        self.visual_teacher = eval(args.model_name)(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, self.norm_adj, edge_index=edges)
        self.id_teacher = eval(args.model_name)(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats, self.norm_adj, edge_index=edges)
        self.textual_teacher.load_state_dict(torch.load('../teacher_models/%s_%s_2' % (args.model_name, args.dataset), map_location='cpu')['state_dict'])
        self.visual_teacher.load_state_dict(torch.load('../teacher_models/%s_%s_3' % (args.model_name, args.dataset), map_location='cpu')['state_dict'])
        self.textual_teacher = self.textual_teacher.cuda()
        self.visual_teacher = self.visual_teacher.cuda()
        for param in self.textual_teacher.parameters():
            param.requires_grad = False
        for param in self.visual_teacher.parameters():
            param.requires_grad = False

        self.textual_user_embeds, self.textual_item_embeds, *rest = self.textual_teacher(training=2)
        self.visual_user_embeds, self.visual_item_embeds, *rest = self.visual_teacher(training=3)


    def save_results(self, ret, infer_type=0):
        with open(self.save_file_name, 'a') as f:
            f.writelines('%s,%s,%s,kd_ratio=%.4f,kl=%.4f,train=%d,test=%d,training_time=%.3f,%.4f,%.4f,%.4f\n' % (self.start_time, args.dataset, args.model_name, args.kd_ratio, args.kl, args.train_type, infer_type, ret['avg_training_time'],
            ret['recall'][1], ret['ndcg'][1], ret['precision'][1]))


    def test(self, users_to_test, is_val, train_type=1):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model(training=train_type)
            result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
            return result
    def train(self):
        training_time_list = []
        stopping_step = 0
        ratio_logger = []
        textual_ratio, visual_ratio = 1, 1

        val_recall_trace, text_val_recall_trace, image_val_recall_trace = [], [], []
        users_to_val = list(data_generator.val_set.keys())
        ret = self.test(users_to_val, is_val=True, train_type=1)
        text_ret = self.test(users_to_val, is_val=True, train_type=2)
        vis_ret = self.test(users_to_val, is_val=True, train_type=3)
        val_recall_trace.append(ret['recall'][1])
        text_val_recall_trace.append(text_ret['recall'][1])
        image_val_recall_trace.append(vis_ret['recall'][1])

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall, best_ndcg = 0, 0
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss, kd_loss = 0., 0., 0., 0., 0.
            visual_loss, textual_loss, id_loss = 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                users, pos_items, neg_items = data_generator.sample()
                ua_embeddings, ia_embeddings = self.model()
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(ua_embeddings[users], ia_embeddings[pos_items], ia_embeddings[neg_items])


                # calculate kd loss
                textual_ua_embeddings, textual_ia_embeddings, *rest = self.model(training=2)
                visual_ua_embeddings, visual_ia_embeddings, *rest = self.model(training=3)
                
                textual_teacher_ranking = torch.mul(self.textual_user_embeds[users], self.textual_item_embeds[pos_items]).sum(dim=-1) - \
                    torch.mul(self.textual_user_embeds[users], self.textual_item_embeds[neg_items]).sum(dim=-1)
                visual_teacher_ranking = torch.mul(self.visual_user_embeds[users], self.visual_item_embeds[pos_items]).sum(dim=-1) - \
                    torch.mul(self.visual_user_embeds[users], self.visual_item_embeds[neg_items]).sum(dim=-1)
                textual_student_ranking = torch.mul(textual_ua_embeddings[users], textual_ia_embeddings[pos_items]).sum(dim=-1) - \
                    torch.mul(textual_ua_embeddings[users], textual_ia_embeddings[neg_items]).sum(dim=-1)
                visual_student_ranking = torch.mul(visual_ua_embeddings[users], visual_ia_embeddings[pos_items]).sum(dim=-1) - \
                    torch.mul(visual_ua_embeddings[users], visual_ia_embeddings[neg_items]).sum(dim=-1)
                textual_kd_loss = torch.clamp(textual_teacher_ranking - textual_student_ranking, 0)
                visual_kd_loss = torch.clamp(visual_teacher_ranking - visual_student_ranking, 0)

                kd_users, kd_items_j, kd_items_k = data_generator.kd_sample(batch_size=5000)
                textual_student_ranking = torch.mul(textual_ua_embeddings[kd_users], textual_ia_embeddings[kd_items_j]).sum(dim=-1) - \
                    torch.mul(textual_ua_embeddings[kd_users], textual_ia_embeddings[kd_items_k]).sum(dim=-1)
                
                visual_student_ranking = torch.mul(visual_ua_embeddings[kd_users], visual_ia_embeddings[kd_items_j]).sum(dim=-1) - \
                    torch.mul(visual_ua_embeddings[kd_users], visual_ia_embeddings[kd_items_k]).sum(dim=-1)
                
                textual_teacher_ranking = torch.mul(self.textual_user_embeds[kd_users], self.textual_item_embeds[kd_items_j]).sum(dim=-1) - \
                    torch.mul(self.textual_user_embeds[kd_users], self.textual_item_embeds[kd_items_k]).sum(dim=-1)
                
                visual_teacher_ranking = torch.mul(self.visual_user_embeds[kd_users], self.visual_item_embeds[kd_items_j]).sum(dim=-1) - \
                    torch.mul(self.visual_user_embeds[kd_users], self.visual_item_embeds[kd_items_k]).sum(dim=-1)
                textual_kd_loss += args.kl * KLDiverge(textual_teacher_ranking, textual_student_ranking, args.pred_temp)
                visual_kd_loss += args.kl * KLDiverge(visual_teacher_ranking, visual_student_ranking, args.pred_temp)

                # calculate kd weight
                joint_sim = (ua_embeddings[users] * ia_embeddings[pos_items]).sum(dim=-1) - \
                        (ua_embeddings[users] * ia_embeddings[neg_items]).sum(dim=-1)
                visual_sim = (visual_ua_embeddings[users] * visual_ia_embeddings[pos_items]).sum(dim=-1) - \
                        (visual_ua_embeddings[users] * visual_ia_embeddings[neg_items]).sum(dim=-1)
                textual_sim = (textual_ua_embeddings[users] * textual_ia_embeddings[pos_items]).sum(dim=-1) - \
                        (textual_ua_embeddings[users] * textual_ia_embeddings[neg_items]).sum(dim=-1)
                coeff_txt_given_vis = torch.clamp((joint_sim - visual_sim) / textual_sim, 1e-8, 10).detach()
                coeff_vis_given_txt = torch.clamp((joint_sim - textual_sim) / visual_sim, 1e-8, 10).detach()
                denom = coeff_txt_given_vis + coeff_vis_given_txt
                textual_ratio = 1 - (coeff_txt_given_vis - coeff_vis_given_txt) / denom
                visual_ratio = 2 - textual_ratio
                batch_kd_loss = textual_ratio * textual_kd_loss + visual_ratio * visual_kd_loss
                batch_kd_loss *= args.kd_ratio


                batch_mf_loss = batch_mf_loss.mean()
                batch_kd_loss = batch_kd_loss.mean()
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_kd_loss 

                batch_loss.backward(retain_graph=True)



                self.optimizer.step()
                
                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                visual_loss += float(visual_kd_loss.mean()) * args.kd_ratio * visual_ratio
                textual_loss += float(textual_kd_loss.mean()) * args.kd_ratio * textual_ratio
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
                kd_loss += float(batch_kd_loss)
                
            training_time_list.append(time() - t1)
            del ua_embeddings, ia_embeddings


            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, kd_loss)
                self.logger.logging(perf_str)
                continue


            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True, train_type=1)
            text_ret = self.test(users_to_val, is_val=True, train_type=2)
            vis_ret = self.test(users_to_val, is_val=True, train_type=3)
            val_recall_trace.append(ret['recall'][1])
            text_val_recall_trace.append(text_ret['recall'][1])
            image_val_recall_trace.append(vis_ret['recall'][1])

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
                best_ndcg = ret['ndcg'][1]
                if args.infer_type == 0:
                    test_ret = self.test(users_to_test, is_val=False)
                    self.logger.logging("Test_Recall@%d: %.5f" % (eval(args.Ks)[1], test_ret['recall'][1]))
                else:
                    test_ret, test_textual_ret, test_visual_ret, test_id_ret = self.test(users_to_test, is_val=False)
                    self.logger.logging("Test_Recall@%d: %.5f  Test_Textual_Recall@%d: %.5f  Test_Visual_Recall@%d: %.5f  Test_ID_Recall@%d: %.5f" % (eval(args.Ks)[1], test_ret['recall'][1], eval(args.Ks)[1], test_textual_ret['recall'][1], eval(args.Ks)[1], test_visual_ret['recall'][1], eval(args.Ks)[1], test_id_ret['recall'][1]))
                stopping_step = 0
                if args.save_model:
                    if not os.path.exists('../kd_models'):
                        os.makedirs('../kd_models')
                    torch.save({'state_dict': self.model.state_dict()} ,'../kd_models/%s_%s_%.4f_%.4f' % (args.model_name, args.dataset, args.kd_ratio, args.kl))
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break

        avg_training_time = np.mean(training_time_list)
        test_ret['avg_training_time'] = avg_training_time
        self.logger.logging(str(test_ret))
        self.save_results(test_ret)
        if args.infer_type == 1:
            self.save_results(test_textual_ret, infer_type=2)
            self.save_results(test_visual_ret, infer_type=3)
            self.save_results(test_id_ret, infer_type=4)


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)


        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -maxi
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

