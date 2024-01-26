import numpy as np
import torch
import ipdb
import torch.nn.functional as F
from torch import Tensor
import math
import os
import io
import copy
import time
import random

class NTXentLoss(torch.nn.Module):#不加标签信息的loss
    def __init__(self, temperature=0.1, eps=1e-6):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, out_1, out_2):
        # out_1: torch.Size([128, 64])
        # out_2: torch.Size([128, 64])
#         ipdb.set_trace()
        out_1 = torch.nn.functional.normalize(out_1, p=2, dim=1)
        out_2 = torch.nn.functional.normalize(out_2, p=2, dim=1)
        
#         ipdb.set_trace()
        out = torch.cat([out_1, out_2], dim=0) # torch.Size([256, 64])

        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature).sum(dim=-1)

        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)

        neg = torch.clamp(neg - row_sub, min=self.eps)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / (neg + self.eps)).mean()

class EarlyStopping_onetower:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, version='SASRec_V3', verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_performance = None
        self.early_stop = False
        self.ndcg_max = None
        self.save_epoch = None
        self.delta = delta
        self.version = version

    def __call__(self, epoch, model, result_path, t_test):

        if self.ndcg_max is None:
            self.ndcg_max = t_test[2]
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
        elif t_test[2] < self.ndcg_max:
            self.counter += 1
            print(f'In the epoch: {epoch}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
            self.counter = 0

    def save_checkpoint(self, epoch, model, result_path, t_test):
        print(f'Validation loss in {epoch} decreased {self.ndcg_max:.4f} --> {t_test[2]:.4f}.  Saving model ...\n')
        with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
            file.write("NDCG@10 in epoch {} decreased {:.4f} --> {:.4f}, the HR@10 is {:.4f}, the AUC is {:.4f}, the loss_rec is {:.4f}, distance_mix_source: {:.4f}, distance_mix_target: {:.4f}, distance_source_target: {:.4f}. Saving model...\n".format(epoch, self.ndcg_max, t_test[2], t_test[7], t_test[10], t_test[11], t_test[12], t_test[13], t_test[14]))
        torch.save(model.state_dict(), os.path.join(result_path, 'checkpoint.pt')) 
        self.ndcg_max = t_test[2]
        
        

    
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
class SASRec_Embedding(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_Embedding, self).__init__()

        self.item_num = item_num # 3416
        self.dev = args.device #'cuda'

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) #Embedding(3417, 50, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE Embedding(200, 50)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) #Dropout(p=0.2)

        self.attention_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.attention_layers = torch.nn.ModuleList() # 2 layers of MultiheadAttention
        self.forward_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.forward_layers = torch.nn.ModuleList() # 2 layers of PointWiseFeedForward

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) #LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate, batch_first=True) # MultiheadAttention((out_proj): NonDynamicallyQuantizableLinear(in_features=50, out_features=50, bias=True))
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm((50,), eps=1e-08, elementwise_affine=True)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5 #torch.Size([128, 200, 50])
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) #(128, 200)

            # add the position embedding
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev)) #torch.Size([128, 200, 50])
        seqs = self.emb_dropout(seqs)

            # mask the noninteracted position
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev) # (128,200)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality, 200
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev)) #(200,200)

        for i in range(len(self.attention_layers)):
#             seqs = torch.transpose(seqs, 0, 1) # torch.Size([200, 128, 50])
            Q = self.attention_layernorms[i](seqs) #torch.Size([128, 200, 50])
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask) # torch.Size([128, 200, 50])
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs # torch.Size([128, 200, 50])
#             seqs = torch.transpose(seqs, 0, 1) # torch.Size([128, 200, 50])

            seqs = self.forward_layernorms[i](seqs) # torch.Size([128, 200, 50])
            seqs = self.forward_layers[i](seqs) # torch.Size([128, 200, 50])
            seqs *=  ~timeline_mask.unsqueeze(-1) # torch.Size([128, 200, 50])

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

    def forward(self, log_seqs): # for training      
            # user_ids:(128,)
            # log_seqs:(128, 200)
            # pos_seqs:(128, 200)
            # neg_seqs:(128, 200)
        log_feats = self.log2feats(log_seqs) # torch.Size([128, 200, 50]) user_ids hasn't been used yet


        return log_feats # pos_pred, neg_pred    
    


class SASRec_V12_time_final(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_V12_time_final, self).__init__()
        
        self.sasrec_embedding_mix = SASRec_Embedding(item_num, args)
        self.sasrec_embedding_source = SASRec_Embedding(item_num, args)
        self.sasrec_embedding_target = SASRec_Embedding(item_num, args)
        self.dev = args.device #'cuda'

        # for the both domain
        self.log_feat_map1 = torch.nn.Linear(args.hidden_units * 3, args.hidden_units)
        self.log_feat_map2 = torch.nn.Linear(args.hidden_units, args.hidden_units)

        # contrastive learning mapping for mix
        self.map_mix_cl1_layer1 = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.map_mix_cl1_layer2 = torch.nn.Linear(args.hidden_units, args.hidden_units)
        
        # contrastive learning mapping for source
        self.map_source_cl1_layer1 = torch.nn.Linear(args.hidden_units, args.hidden_units)
        self.map_source_cl1_layer2 = torch.nn.Linear(args.hidden_units, args.hidden_units)
        
        # contrastive learning mapping for target
        self.map_target_cl1_layer1 = torch.nn.Linear(args.hidden_units, args.hidden_units)   
        self.map_target_cl1_layer2 = torch.nn.Linear(args.hidden_units, args.hidden_units)   


        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=2)
        self.temperature = args.temperature
        self.fname = args.dataset

        # Attention Layer_new 2022.09.27
        layers = []
        layers.append(torch.nn.Linear(args.hidden_units * 5, 80))
        layers.append(torch.nn.PReLU())
        layers.append(torch.nn.Linear(80, 40))
        layers.append(torch.nn.PReLU())
        layers.append(torch.nn.Linear(40, 1))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, user_ids, mix_log_seqs, source_log_seqs, target_log_seqs, pos_seqs, neg_seqs, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices): # for training     
#         ipdb.set_trace()
        if self.fname == 'amazon_game':
            mix_log_feats_sas = self.sasrec_embedding_mix(mix_log_seqs) # torch.Size([128, 200, 64])
            source_log_feats_sas = self.sasrec_embedding_source(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats_sas = self.sasrec_embedding_target(target_log_seqs) # torch.Size([128, 200, 64])

#             ipdb.set_trace()
            start_index_target = min(np.where(target_log_seqs != 0)[1])
            if start_index_target == 0:
#                 ipdb.set_trace()
                mix_log_feats = mix_log_feats_sas[:, 0, :].unsqueeze(1)
                source_log_feats = source_log_feats_sas[:, 0, :].unsqueeze(1)
                target_log_feats = target_log_feats_sas[:, 0, :].unsqueeze(1)
                for i in range(1, target_log_seqs.shape[1]):
#                     ipdb.set_trace()
                    matrix_mix, end_position_mix = self.get_mask_matrix(mix_log_seqs, user_train_mix_sequence_for_target_indices, i)
                    matrix_source, end_position_source = self.get_mask_matrix(source_log_seqs, user_train_source_sequence_for_target_indices, i)
                    mix_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], mix_log_feats_sas, matrix_mix.cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    source_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], source_log_feats_sas, matrix_source.cuda(), mix_log_feats_sas, matrix_mix.cuda())
        
                    target_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], target_log_feats_sas[:, 0:i+1, :], torch.tensor(target_log_seqs[:, 0:i+1]).cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    mix_log_feats = torch.cat([mix_log_feats, mix_log_feats_final_dim.unsqueeze(1)], dim=1)
                    source_log_feats = torch.cat([source_log_feats, source_log_feats_final_dim.unsqueeze(1)], dim=1)
                    target_log_feats = torch.cat([target_log_feats, target_log_feats_final_dim.unsqueeze(1)], dim=1)
            else:
                mix_log_feats = mix_log_feats_sas[:, 0:start_index_target, :]
                source_log_feats = source_log_feats_sas[:, 0:start_index_target, :]
                target_log_feats = target_log_feats_sas[:, 0:start_index_target, :]
                for i in range(start_index_target, target_log_seqs.shape[1]):
#                     ipdb.set_trace()
                    matrix_mix, end_position_mix = self.get_mask_matrix(mix_log_seqs, user_train_mix_sequence_for_target_indices, i)
                    matrix_source, end_position_source = self.get_mask_matrix(source_log_seqs, user_train_source_sequence_for_target_indices, i)
                    mix_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], mix_log_feats_sas, matrix_mix.cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    source_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], source_log_feats_sas, matrix_source.cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    target_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], target_log_feats_sas[:, start_index_target:i+1, :], torch.tensor(target_log_seqs[:, start_index_target:i+1]).cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    mix_log_feats = torch.cat([mix_log_feats, mix_log_feats_final_dim.unsqueeze(1)], dim=1)
                    source_log_feats = torch.cat([source_log_feats, source_log_feats_final_dim.unsqueeze(1)], dim=1)
                    target_log_feats = torch.cat([target_log_feats, target_log_feats_final_dim.unsqueeze(1)], dim=1)
            
            concatenate_log_feats = torch.cat([mix_log_feats, source_log_feats, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats))) # torch.Size([128, 200, 64])

            pos_embs = self.sasrec_embedding_target.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) 
            neg_embs = self.sasrec_embedding_target.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) 
        elif self.fname == 'amazon_toy':
            mix_log_feats_sas = self.sasrec_embedding_mix(mix_log_seqs) # torch.Size([128, 200, 64])
            source_log_feats_sas = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats_sas = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])
            
#             ipdb.set_trace()
            start_index_target = min(np.where(target_log_seqs != 0)[1])
            if start_index_target == 0:
#                 ipdb.set_trace()
                mix_log_feats = mix_log_feats_sas[:, 0, :].unsqueeze(1)
                source_log_feats = source_log_feats_sas[:, 0, :].unsqueeze(1)
                target_log_feats = target_log_feats_sas[:, 0, :].unsqueeze(1)
                for i in range(1, target_log_seqs.shape[1]):
#                     ipdb.set_trace()
                    matrix_mix, end_position_mix = self.get_mask_matrix(mix_log_seqs, user_train_mix_sequence_for_target_indices, i)
                    matrix_source, end_position_source = self.get_mask_matrix(source_log_seqs, user_train_source_sequence_for_target_indices, i)
                    mix_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], mix_log_feats_sas, matrix_mix.cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    source_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], source_log_feats_sas, matrix_source.cuda(), mix_log_feats_sas, matrix_mix.cuda())
        
                    target_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], target_log_feats_sas[:, 0:i+1, :], torch.tensor(target_log_seqs[:, 0:i+1]).cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    mix_log_feats = torch.cat([mix_log_feats, mix_log_feats_final_dim.unsqueeze(1)], dim=1)
                    source_log_feats = torch.cat([source_log_feats, source_log_feats_final_dim.unsqueeze(1)], dim=1)
                    target_log_feats = torch.cat([target_log_feats, target_log_feats_final_dim.unsqueeze(1)], dim=1)
            else:
                mix_log_feats = mix_log_feats_sas[:, 0:start_index_target, :]
                source_log_feats = source_log_feats_sas[:, 0:start_index_target, :]
                target_log_feats = target_log_feats_sas[:, 0:start_index_target, :]
                for i in range(start_index_target, target_log_seqs.shape[1]):
#                     ipdb.set_trace()
                    matrix_mix, end_position_mix = self.get_mask_matrix(mix_log_seqs, user_train_mix_sequence_for_target_indices, i)
                    matrix_source, end_position_source = self.get_mask_matrix(source_log_seqs, user_train_source_sequence_for_target_indices, i)
                    mix_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], mix_log_feats_sas, matrix_mix.cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    source_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], source_log_feats_sas, matrix_source.cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    target_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, i, :], target_log_feats_sas[:, start_index_target:i+1, :], torch.tensor(target_log_seqs[:, start_index_target:i+1]).cuda(), mix_log_feats_sas, matrix_mix.cuda())
                    mix_log_feats = torch.cat([mix_log_feats, mix_log_feats_final_dim.unsqueeze(1)], dim=1)
                    source_log_feats = torch.cat([source_log_feats, source_log_feats_final_dim.unsqueeze(1)], dim=1)
                    target_log_feats = torch.cat([target_log_feats, target_log_feats_final_dim.unsqueeze(1)], dim=1)
            
            
            concatenate_log_feats = torch.cat([mix_log_feats, source_log_feats, target_log_feats], dim=2)
            log_feats = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            pos_embs = self.sasrec_embedding_source.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # torch.Size([128, 200, 64])
            neg_embs = self.sasrec_embedding_source.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # torch.Size([128, 200, 64])

        # get the l2 norm for the both domains recommendation
        log_feats_l2norm = torch.nn.functional.normalize(log_feats, p=2, dim=2)
        pos_embs_l2norm = torch.nn.functional.normalize(pos_embs, p=2, dim=2)
        neg_embs_l2norm = torch.nn.functional.normalize(neg_embs, p=2, dim=2)
        pos_logits = (log_feats_l2norm * pos_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        neg_logits = (log_feats_l2norm * neg_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        
        pos_logits = pos_logits * self.temperature
        neg_logits = neg_logits * self.temperature

        mix_last_feat = self.map_mix_cl1_layer2(self.relu(self.map_mix_cl1_layer1(mix_log_feats[:, -1, :])))
        source_last_feat = self.map_source_cl1_layer2(self.relu(self.map_source_cl1_layer1(source_log_feats[:, -1, :])))
        target_last_feat = self.map_target_cl1_layer2(self.relu(self.map_target_cl1_layer1(target_log_feats[:, -1, :])))
        return mix_last_feat, source_last_feat, target_last_feat, pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, mix_log_seqs, source_log_seqs, target_log_seqs, item_indices): # for inference
            # user_ids: (1,)
            # log_seqs: (1, 200)
            # item_indices: (101,)e
        if self.fname == 'amazon_game':
            mix_log_feats_sas = self.sasrec_embedding_mix(mix_log_seqs) # torch.Size([128, 200, 64])
            source_log_feats_sas = self.sasrec_embedding_source(source_log_seqs) # torch.Size([1, 200, 50]) 
            target_log_feats_sas = self.sasrec_embedding_target(target_log_seqs) # torch.Size([1, 200, 50]) 
#             ipdb.set_trace()
#             # Attention for the last dimension
                # Attention in the mix domain for the last dimension
            position = 200
            mix_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, position-1, :], mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda(), mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda())
                # Attention in the source domain for the last dimension
            source_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, position-1, :], source_log_feats_sas, torch.tensor(source_log_seqs).cuda(), mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda())
                # Attention in the target domain for the last dimension
            target_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, position-1, :], target_log_feats_sas, torch.tensor(target_log_seqs).cuda(), mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda())

            concatenate_log_feats = torch.cat([mix_log_feats_final_dim, source_log_feats_final_dim, target_log_feats_final_dim], dim=1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))

            item_embs = self.sasrec_embedding_target.item_emb(torch.LongTensor(item_indices).to(self.dev)) 

        elif self.fname == 'amazon_toy':
            mix_log_feats_sas = self.sasrec_embedding_mix(mix_log_seqs) # torch.Size([128, 200, 64])
            source_log_feats_sas = self.sasrec_embedding_target(source_log_seqs) # torch.Size([128, 200, 64])
            target_log_feats_sas = self.sasrec_embedding_source(target_log_seqs) # torch.Size([128, 200, 64])
            
            # Attention for the last dimension
            position = 200
                # Attention in the source domain for the last dimension
            mix_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, position-1, :], mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda(), mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda())
                # Attention in the source domain for the last dimension
            source_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, position-1, :], source_log_feats_sas, torch.tensor(source_log_seqs).cuda(), mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda())
                # Attention in the target domain for the last dimension
            target_log_feats_final_dim = self.DINAttentionLayer_withmixattn(target_log_feats_sas[:, position-1, :], target_log_feats_sas, torch.tensor(target_log_seqs).cuda(), mix_log_feats_sas, torch.tensor(mix_log_seqs).cuda())
            
            
            concatenate_log_feats = torch.cat([mix_log_feats_final_dim, source_log_feats_final_dim, target_log_feats_final_dim], dim=1)
            final_feat = self.log_feat_map2(self.leakyrelu(self.log_feat_map1(concatenate_log_feats)))
            
            item_embs = self.sasrec_embedding_source.item_emb(torch.LongTensor(item_indices).to(self.dev)) 
            
            
        # get the l2 norm for the both domains recommendation
#         final_feat = log_feats[:, -1, :] # torch.Size([1, 50]) 
        final_feat_l2norm = torch.nn.functional.normalize(final_feat, p=2, dim=1)
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=1)
        
        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) 
        logits = logits * self.temperature
        
        mix_last_feat = self.map_mix_cl1_layer2(self.relu(self.map_mix_cl1_layer1(mix_log_feats_final_dim)))
        source_last_feat = self.map_source_cl1_layer2(self.relu(self.map_source_cl1_layer1(source_log_feats_final_dim)))
        target_last_feat = self.map_target_cl1_layer2(self.relu(self.map_target_cl1_layer1(target_log_feats_final_dim)))
        
        return logits, mix_last_feat, source_last_feat, target_last_feat # preds # (U, I)
        
    # Attention function_new 2022.09.27
    def DINAttentionLayer_withmixattn(self, query, fact, mask, mix_fact, mix_mask):
        # query: torch.Size([128, 36])
        # fact: torch.Size([128, 100, 36])
        # mask: torch.Size([128, 100])
        B, T, D = fact.shape # 128, 100, 36
        query_full = torch.tile(query.unsqueeze(1), (1, T, 1)) # torch.Size([128, 100, 36])
#         ipdb.set_trace()
#         mix_weight = torch.where(mix_mask != 0, torch.tile(torch.arange(0,mix_mask.shape[1],dtype=torch.float32).unsqueeze(0), [mix_mask.shape[0], 1]).cuda(), torch.ones_like(mix_mask) * (-2 ** 31)) # torch.Size([64, 200])
        mix_weight = torch.where(mix_mask != 0, torch.ones_like(mix_mask,dtype=torch.float32), torch.ones_like(mix_mask) * (-2 ** 31)) # torch.Size([64, 200])
        mix_weight = mix_weight.softmax(dim = -1).view((mix_mask.shape[0] , 1, mix_mask.shape[1])) # torch.Size([64, 1, 200])
        
        mix_attn = torch.tile(torch.matmul(mix_weight, mix_fact), (1, T, 1))
        
        combination = torch.cat([fact, query_full, fact * query_full, query_full - fact, mix_attn], dim = 2) # torch.Size([64, 200, 320])

        scores = self.model(combination).squeeze(-1) # torch.Size([64, 200])
        scores = torch.where(mask != 0, scores, torch.ones_like(scores) * (-2 ** 31)) # torch.Size([64, 200])
        scores = scores.softmax(dim = -1).view((B , 1, T)) # torch.Size([64, 1, 200])
#         prob = copy.deepcopy(scores)

        return torch.matmul(scores, fact).squeeze(1) # torch.Size([128, 36])

    def get_mask_matrix(self, log_seqs, sequence_for_target_indices, column):
#         ipdb.set_trace()
        column_position = torch.tensor(sequence_for_target_indices[:,column]).unsqueeze(1)
        position = torch.tile(torch.arange(0,log_seqs.shape[1]).unsqueeze(0), [log_seqs.shape[0], 1])
        position_mask_matrix = torch.where(position > column_position, torch.zeros_like(position), torch.ones_like(position))
        mask_matrix = torch.tensor(log_seqs) * position_mask_matrix
        end_position_list = [0]
        end_position_list.extend(torch.nonzero(mask_matrix)[:,1].numpy().tolist())
        end_position = max(end_position_list)
        return mask_matrix.detach(), end_position