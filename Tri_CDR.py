import os
import time
import torch
import argparse
import ipdb

from model import SASRec_V12_time_final
from model import EarlyStopping_onetower
from model import NTXentLoss

from utils import *
import os
import io

from matplotlib.pyplot import MultipleLocator

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# -*- coding: UTF-8 -*-
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=2000)

from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
    
# load weights
def get_updateModel(model, path_mix, path_source, path_target):

    pretrained_dict_mix = torch.load(path_mix, map_location='cpu') # 68
    pretrained_dict_source = torch.load(path_source, map_location='cpu') # 68
    pretrained_dict_target = torch.load(path_target, map_location='cpu') # 68
    model_dict = model.state_dict() # 68
    
    shared_dict_mix = {k: v for k, v in pretrained_dict_mix.items() if k.startswith('sasrec_embedding_mix')}# 28
    shared_dict_source = {k: v for k, v in pretrained_dict_source.items() if k.startswith('sasrec_embedding_source')}# 28
    shared_dict_target = {k: v for k, v in pretrained_dict_target.items() if k.startswith('sasrec_embedding_target')}# 28

    model_dict.update(shared_dict_mix)
    model_dict.update(shared_dict_source)
    model_dict.update(shared_dict_target)
    
    print("Load the length of mix is:", len(shared_dict_mix.keys()))
    print("Load the length of source is:", len(shared_dict_source.keys()))
    print("Load the length of target is:", len(shared_dict_target.keys()))

    model.load_state_dict(model_dict)
    return model
    

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--cross_dataset', required=True)
parser.add_argument('--batch_size', default=120, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--num_samples', default=100, type=int)
parser.add_argument('--decay', default=4, type=int)
parser.add_argument('--lr_decay_rate', default=0.99, type=float)
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--version', default=None, type=str)
parser.add_argument('--lr_linear', default=0.01, type=float)
parser.add_argument('--start_decay_linear', default=8, type=int)
parser.add_argument('--temperature', default=5, type=float)
parser.add_argument('--seed', default=5, type=int)
parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--info_NCE_temperature', default=0.1, type=float)
parser.add_argument('--rec_ratio_cl1', default=2.0, type=float)
parser.add_argument('--rate_mix_source', default=1.0, type=float)
parser.add_argument('--rate_mix_target', default=1.0, type=float)
parser.add_argument('--rate_source_target', default=1.0, type=float)
parser.add_argument('--cl_weight', default=1.0, type=float)
parser.add_argument('--triplet_weight', default=1.0, type=float)
parser.add_argument('--triplet_margin', default=1.0, type=float)

args = parser.parse_args()


SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)




result_path = './Log_File_' + str(args.dataset) + '/Tri-CDR/'
print("Save in path:", result_path)
if not os.path.isdir(result_path):
    os.makedirs(result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

if args.cross_dataset == 'Book_Movie':
    source_name = 'book'
    target_name = 'movie'
elif args.cross_dataset == 'Toy_Game':
    source_name = 'toy'
    target_name = 'game'

rate_for_mix_source = args.rate_mix_source / (args.rate_mix_source + args.rate_mix_target + args.rate_source_target)
rate_for_mix_target = args.rate_mix_target / (args.rate_mix_source + args.rate_mix_target + args.rate_source_target)
rate_for_source_target = args.rate_source_target / (args.rate_mix_source + args.rate_mix_target + args.rate_source_target)
print("the rate between mix and source:", rate_for_mix_source)
print("the rate between mix and target:", rate_for_mix_target)
print("the rate between source and target:", rate_for_source_target)

        
if __name__ == '__main__':
    # global dataset
#     ipdb.set_trace()
#     print(os.getcwd())
    dataset = data_partition(args.version, args.dataset, args.cross_dataset, args.maxlen)

    [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval] = dataset
#     [user_train_source, user_train_target, user_valid_source, user_valid_target, user_test_source, user_test_target, usernum, itemnum, interval] = dataset
    num_batch = len(user_train_source) // args.batch_size # 908
    cc_source = 0.0
    cc_target = 0.0
    for u in user_train_source:
        cc_source = cc_source + len(user_train_source[u])
        cc_target = cc_target + len(user_train_target[u])
    print('average sequence length in source domain: %.2f' % (cc_source / len(user_train_source)))
    print('average sequence length in target domain: %.2f' % (cc_target / len(user_train_source)))
    print('average sequence length in both domain: %.2f' % ((cc_source + cc_target) / len(user_train_source)))

    sampler = WarpSampler(args.version, args.dataset, args.cross_dataset, interval, user_train_mix, user_train_source, user_train_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, None, None, SEED, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec_V12_time_final(usernum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    if args.cross_dataset == 'Toy_Game':
        toy_model_path = './Checkpoints/SASRec_checkpoint_Toy.pt'
        game_model_path = './Checkpoints/SASRec_checkpoint_Game.pt'
        if args.dataset == 'amazon_toy':
            mix_model_path = './Checkpoints/SASRec_checkpoint_Toy_Mix.pt'
        elif args.dataset == 'amazon_game':
            mix_model_path = './Checkpoints/SASRec_checkpoint_Game_Mix.pt'
#         ipdb.set_trace()
        get_updateModel(model, mix_model_path, toy_model_path, game_model_path)

    model.train() # enable model training
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    cl_criterion = NTXentLoss(temperature = args.info_NCE_temperature)
    triplet_criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # set the early stop
    early_stopping = EarlyStopping_onetower(args.patience, version='SASRec_V3', verbose=True) 

    # set the learning rate scheduler
    if args.lrscheduler == 'Steplr': # 
        learningrate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=args.decay, gamma=args.lr_decay_rate, verbose=True)
    elif args.lrscheduler == 'ExponentialLR': # 
        learningrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=args.lr_decay_rate, last_epoch=-1, verbose=True)
    elif args.lrscheduler == 'CosineAnnealingLR':
        learningrate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1, verbose=True)
    
    T = 0.0
    t0 = time.time()
    epoch_list = []
    lr_list = []
    loss_train_rec_list = []
    loss_train_cl_list = []
    loss_train_triplet_list = []
    loss_train_list = []
    loss_test_list = []
    ndcg_list = []
    hr_list = []
    auc_list = []
    cl_weight_num = args.cl_weight
    triplet_weight_num = args.triplet_weight
    print("The weight of CL is {}, and the weight of Traiplrt is {}".format(cl_weight_num, triplet_weight_num))
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_list.append(epoch)
        lr_list.append(learningrate_scheduler.get_last_lr())

        t1 = time.time()
        loss_mix_source = 0
        loss_mix_target = 0
        loss_source_target = 0
        loss_rec_epoch = 0
        loss_cl1_epoch = 0
        loss_triplet_epoch = 0
        loss_epoch = 0
        distance_mix_source = 0
        distance_mix_target = 0
        distance_source_target = 0
#         lr_scheduler(epoch, args)
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
# #             ipdb.set_trace()
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = sampler.next_batch() # tuples to ndarray
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices = np.array(u), np.array(seq_mix), np.array(seq_source), np.array(seq_target), np.array(pos_target), np.array(neg_target), np.array(user_train_mix_sequence_for_target_indices), np.array(user_train_source_sequence_for_target_indices)        
#             ipdb.set_trace()
            mix_log_feats, source_log_feats, target_log_feats, pos_logits, neg_logits = model(u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos_target != 0)
            loss_rec = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss_rec += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            cl_loss_mix_source = cl_criterion(mix_log_feats, source_log_feats)
            cl_loss_mix_target = cl_criterion(mix_log_feats, target_log_feats)
            cl_loss_source_target = cl_criterion(source_log_feats, target_log_feats)
            loss_cl1 = cl_loss_mix_source * rate_for_mix_source + cl_loss_mix_target * rate_for_mix_target + cl_loss_source_target * rate_for_source_target

            distance_mix_source_batch = torch.dist(mix_log_feats, source_log_feats, p=2)
            distance_mix_target_batch = torch.dist(mix_log_feats, target_log_feats, p=2)
            distance_source_target_batch = torch.dist(source_log_feats, target_log_feats, p=2)
            
            distance_mix_source += distance_mix_source_batch.item()
            distance_mix_target += distance_mix_target_batch.item()
            distance_source_target += distance_source_target_batch.item()

            loss_triplet = triplet_criterion(source_log_feats, mix_log_feats, target_log_feats)
#             ipdb.set_trace()  
            loss = loss_rec + loss_cl1 * cl_weight_num + loss_triplet * triplet_weight_num
            loss_rec_epoch += loss_rec.item()
            loss_cl1_epoch += loss_cl1.item() * cl_weight_num
            loss_triplet_epoch += loss_triplet.item() * triplet_weight_num
            loss_epoch += loss.item()
            
#             for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
#             ipdb.set_trace()
            print("In epoch {} iteration {}: loss_rec={:.4f}, loss_cl1={:.4f}, loss_triplet={:.4f}, loss_all={:.4f}".format(epoch, step, loss_rec.item(), loss_cl1.item() * cl_weight_num, loss_triplet.item() * triplet_weight_num, loss.item())) 
            with io.open(result_path + 'loss_log.txt', 'a', encoding='utf-8') as file:
                file.write("In epoch {} iteration {}: loss_rec={:.4f}, loss_cl1={:.4f}, loss_triplet={:.4f}, loss_all={:.4f}\n".format(epoch, step, loss_rec.item(), loss_cl1.item() * cl_weight_num, loss_triplet.item() * triplet_weight_num, loss.item()))
        loss_train_rec_list.append(loss_rec_epoch / num_batch)
        loss_train_cl_list.append(loss_cl1_epoch / num_batch)
        loss_train_triplet_list.append(loss_triplet_epoch / num_batch)
        loss_train_list.append(loss_epoch / num_batch)
        learningrate_scheduler.step()
        
        with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write("In epoch {}: loss_rec={}, loss_cl1={}, loss_triplet={}, loss_all={}, time: {}\n".format(epoch, loss_rec_epoch / num_batch, loss_cl1_epoch / num_batch, loss_triplet_epoch / num_batch, loss_epoch / num_batch, time.time() - t1))
            
        model.eval()
        T = time.time() - t0
        t_test = evaluate_SASRec(model, dataset, args)    
        print('epoch:%d, total_time: %f(s), test:    NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f, distance_mix_source: %.4f, distance_mix_target: %.4f, distance_source_target: %.4f\n' % (epoch, T, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11], t_test[12], t_test[13], t_test[14]))

        
        ndcg_list.append(t_test[2])
        hr_list.append(t_test[7])
        auc_list.append(t_test[10])
        loss_test_list.append(t_test[11])


            
        with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
            file.write('epoch:%d, time: %f(s), test:    NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f, distance_mix_source: %.4f, distance_mix_target: %.4f, distance_source_target: %.4f\n' % (epoch, T, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11], t_test[12], t_test[13], t_test[14]))
        model.train()
        
        early_stopping(epoch, model, result_path, t_test)
        if early_stopping.early_stop:
            print("Save in path:{}\n".format(result_path))
            print("Early stopping in the epoch {}, the NDCG@10: {:.4f}, HR@10: {:.4f}, AUC: {:.4f}, loss: {:.4f}, distance_mix_source: {:.4f}, distance_mix_target: {:.4f}, distance_source_target: {:.4f}\n".format(early_stopping.save_epoch, early_stopping.best_performance[2], early_stopping.best_performance[7], early_stopping.best_performance[10], early_stopping.best_performance[11], early_stopping.best_performance[12], early_stopping.best_performance[13], early_stopping.best_performance[14]))
            with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
                file.write("Early stopping in the epoch {}, the NDCG@10: {:.4f}, HR@10: {:.4f}, AUC: {:.4f}, loss_rec: {:.4f}, distance_mix_source: {:.4f}, distance_mix_target: {:.4f}, distance_source_target: {:.4f}\n".format(epoch, early_stopping.best_performance[2], early_stopping.best_performance[7], early_stopping.best_performance[10], early_stopping.best_performance[11], early_stopping.best_performance[12], early_stopping.best_performance[13], early_stopping.best_performance[14]))
            break

    
    sampler.close()

    plt.figure(figsize=(15, 8)) 
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.step(epoch_list, lr_list, "green", marker="8", markersize=5, label="lr")
    plt.legend(loc='upper right')
    plt.savefig(result_path + 'lr_plot.png')
    plt.close()

#     ipdb.set_trace()

    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.plot(epoch_list, loss_train_rec_list, "black", marker="o", markersize=5, label="loss_train_rec")
    plt.plot(epoch_list, loss_train_cl_list, "green", marker="<", markersize=5, label="loss_train_cl")
    plt.plot(epoch_list, loss_train_triplet_list, "red", marker="v", markersize=5, label="loss_train_triplet")
    plt.plot(epoch_list, loss_train_list, "gray", marker=">", markersize=5, label="loss_train_all")
    plt.plot(epoch_list, loss_test_list, "red", marker="v", markersize=5, label="loss_test_rec")
    plt.plot(epoch_list, ndcg_list, "blue", marker="X", markersize=5, label="ndcg_test")
    plt.plot(epoch_list, hr_list, "yellow", marker="s", markersize=5, label="hr_test")
    plt.plot(epoch_list, auc_list, "orange", marker="P", markersize=5, label="auc_test")
    plt.legend(loc='upper right')
    plt.savefig(result_path + 'performance_plot.png')
    plt.close()
