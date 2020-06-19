'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from tqdm import trange

from utility.helper import *
from utility.batch_test_pytorch import *


class NGCF(nn.Module):
    def __init__(self, data_config):
        super(NGCF, self).__init__()
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.sparse_adj = self._convert_sp_mat_to_sp_tensor(data_config['self_loop_adj']).cuda()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.mess_dropout = eval(args.mess_dropout)
        self.node_dropout = eval(args.node_dropout)[0]

        self.verbose = args.verbose

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.gcn_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        for l in range(self.n_layers):
            self.gcn_layers.append(nn.Linear(self.weight_size[0],self.weight_size[0]))
            self.interaction_layers.append(nn.Linear(self.weight_size[0], self.weight_size[0]))

        # initialization of model parameters
        self.weights = self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for l in range(self.n_layers):
            nn.init.xavier_uniform_(self.gcn_layers[l].weight)
            nn.init.zeros_(self.gcn_layers[l].bias)
            nn.init.xavier_uniform_(self.interaction_layers[l].weight)
            nn.init.zeros_(self.interaction_layers[l].bias)
        print('using xavier initialization')

    def _split_A_hat_node_dropout(self, X):
        noise_shape = X._nnz()
        random_tensor = 1 - self.node_dropout
        random_tensor += torch.rand(noise_shape).cuda()
        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        i = X._indices()[:, dropout_mask]
        v = X._values()[dropout_mask]*(1.0/(1 - self.node_dropout))

        return torch.sparse.FloatTensor(i,v,X.shape).cuda()

    def forward(self, users, pos_items, neg_items, node_drop_flag):
        # Generate a set of adjacency sub-matrix.
        if node_drop_flag:
            # node dropout.
            A_hat = self._split_A_hat_node_dropout(self.sparse_adj)
        else:
            A_hat = self.sparse_adj
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            # sum messages of neighbors.
            side_embeddings = torch.sparse.mm(A_hat,ego_embeddings)
            # transformed sum messages of neighbors.
            sum_embeddings = self.gcn_layers[k](side_embeddings)

            # bi messages of neighbors.
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = self.interaction_layers[k](bi_embeddings)

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = nn.Dropout(1 - self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings[users], i_g_embeddings[pos_items], i_g_embeddings[neg_items]

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        # bpr loss
        bpr_loss = -(pos_scores - neg_scores).sigmoid().log().sum()

        return bpr_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        return torch.sparse.FloatTensor(indices, torch.from_numpy(coo.data).float(), coo.shape)

    def batch_ratings(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, self_loop_adj = data_generator.get_adj_mat()
    config['adj'] = plain_adj
    config['self_loop_adj'] = self_loop_adj

    t0 = time()

    model = NGCF(data_config=config)
    model.cuda()

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    pretrain_path = args.pretrain_path
    if args.pretrain == 1:
        print('with pretrain')
        print('load the pretrained model parameters from: ', pretrain_path)
        model.load_state_dict(torch.load(pretrain_path))
        # *********************************************************
        # get the performance from pretrained model.
        users_to_test = list(data_generator.test_set.keys())
        model.eval()
        ret = test(model, users_to_test, drop_flag=False)
        cur_best_pre_0 = ret['recall'][0]

        pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                       'ndcg=[%.5f, %.5f]' % \
                       (ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1],
                        ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
        print(pretrain_ret)
        exit()
    else:
        print('without pretraining.')

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.regs[0])
    best_recall = 0.

    for epoch in range(args.epoch):
        t1 = time()
        loss = 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        model.train()
        for idx in trange(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            users_embed, pos_items_embed, neg_items_embed = model(users, pos_items, neg_items, args.node_dropout_flag)
            batch_loss = model.create_bpr_loss(users_embed, pos_items_embed, neg_items_embed)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        model.eval()
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f], recall=[%.5f], ' \
                       'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, ret['recall'][0],
                        ret['precision'][0], ret['hit_ratio'][0],
                        ret['ndcg'][0])
            print(perf_str)
        if best_recall < ret['recall'][0]:
            best_recall = ret['recall'][0]
            """
            *********************************************************
            Save the model parameters.
            """
            if args.save_flag:
                print('Saving model at %s' % pretrain_path)
                torch.save(model.state_dict(), pretrain_path)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=50)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
