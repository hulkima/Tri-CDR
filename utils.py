import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import ipdb
import dill as pkl
import time
from sklearn.metrics import roc_auc_score

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
            
# source:book----range[1,interval+1);target:movie[interval+1, itemnum + 1)
def sample_function(version, fname, crossdataset, interval, user_train_mix, user_train_source, user_train_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, batch_size, maxlen, result_queue, SEED):        

    if fname == 'amazon_toy':
        random_min = 1
        random_max = interval + 1
        print("The min is {} and the max is {} in amazon_toy".format(random_min, random_max))
    elif fname == 'amazon_game':
        random_min = interval + 1
        random_max = itemnum + 1
        print("The min is {} and the max is {} in amazon_game".format(random_min, random_max))
        
        
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train_mix[user]) <= 1 or len(user_train_source[user]) <= 1 or len(user_train_target[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)
            
        seq_mix = np.zeros([maxlen], dtype=np.int32)
        seq_source = np.zeros([maxlen], dtype=np.int32)
        seq_target = np.zeros([maxlen], dtype=np.int32)
        pos_target = np.zeros([maxlen], dtype=np.int32)
        neg_target = np.zeros([maxlen], dtype=np.int32)
        user_train_mix_sequence_for_target_indices = np.zeros([maxlen], dtype=np.int32)
        user_train_source_sequence_for_target_indices = np.zeros([maxlen], dtype=np.int32)

        nxt_target = user_train_target[user][-1] # # 最后一个交互的物品

        idx_mix = maxlen - 1 #49
        idx_source = maxlen - 1 #49
        idx_target = maxlen - 1 #49
        
        ts_target = set(user_train_target[user]) # a set
        for i in reversed(range(0, len(user_train_mix[user]))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_mix[idx_mix] = user_train_mix[user][i]
            idx_mix -= 1
            if idx_mix == -1: break
                
        for i in reversed(range(0, len(user_train_source[user]))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_source[idx_source] = user_train_source[user][i]
            idx_source -= 1
            if idx_source == -1: break
                
        for i in reversed(range(0, len(user_train_target[user][:-1]))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_target[idx_target] = user_train_target[user][i]
            pos_target[idx_target] = nxt_target
            if user_train_mix_sequence_for_target[user][i] < -maxlen:
                user_train_mix_sequence_for_target_indices[idx_target] = 0
            else:
                user_train_mix_sequence_for_target_indices[idx_target] = user_train_mix_sequence_for_target[user][i] + maxlen
                
            if user_train_source_sequence_for_target[user][i] < -maxlen or user_train_source_sequence_for_target[user][i] == -len(user_train_source[user])-1:
                user_train_source_sequence_for_target_indices[idx_target] = 0
            else:
                user_train_source_sequence_for_target_indices[idx_target] = user_train_source_sequence_for_target[user][i] + maxlen
            if nxt_target != 0: neg_target[idx_target] = random_neq(random_min, random_max, ts_target)
            nxt_target = user_train_target[user][i]
            idx_target -= 1
            if idx_target == -1: break

        return (user, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices)


    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))
            
            
            
            
            
            
            
            
            
            
class WarpSampler(object):
    def __init__(self, version, fname, crossdataset, interval, user_train_mix, user_train_source, user_train_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, itemnum_source, itemnum_target, SEED, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(version, 
                                                         fname,
                                                         crossdataset,
                                                         interval,
                                                         user_train_mix,
                                                         user_train_source,
                                                         user_train_target,
                                                         user_train_mix_sequence_for_target,
                                                         user_train_source_sequence_for_target, 
                                                         usernum,
                                                         itemnum,
                                                         batch_size,
                                                         maxlen,
                                                         self.result_queue,
                                                         SEED
                                                         )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()   
            
            
            
          
        

        

def data_partition(version, fname, dataset_name, maxlen):
    usernum = 0
    itemnum = 0
    user_train = {}
    user_valid = {}
    user_test = {}
    interval = 0

    with open('./Dateset/toy_log_file_final.pkl', 'rb') as f:
        toy_log_file_final = pkl.load(f)

    with open('./Dateset/game_log_file_final.pkl', 'rb') as f:
        game_log_file_final = pkl.load(f)

    with open('./Dateset/mix_log_file_final.pkl', 'rb') as f:
        mix_log_file_final = pkl.load(f)

    with open('./Dateset/item_index_toy.pkl', 'rb') as f:
        item_index_toy = pkl.load(f)

    with open('./Dateset/item_index_game.pkl', 'rb') as f:
        item_index_game = pkl.load(f)

    with open('./Dateset/item_index_mix.pkl', 'rb') as f:
        item_index_mix = pkl.load(f)

    with open('./Dateset/user_index_overleap.pkl', 'rb') as f:
        user_index_overleap = pkl.load(f)

    item_index_game_array = np.load('./Dateset/item_index_game.npy')
    item_index_toy_array = np.load('./Dateset/item_index_toy.npy')

    interval = 37868

    usernum = len(user_index_overleap.keys()) # 116254
    if fname == 'amazon_game':
        user_train_game_mix = {}
        user_train_game_source = {}
        user_train_game_target = {}
        user_valid_game_target = {}
        user_test_game_target = {}
        user_train_game_mix_sequence_for_target = {}
        user_train_game_source_sequence_for_target = {}
        
        position_mix = []
        position_source = []

        itemnum = len(item_index_mix.keys())
        for k in range(1, len(user_index_overleap.keys()) + 1):
            v_mix_game = copy.deepcopy(mix_log_file_final[k])
            v_game = copy.deepcopy(game_log_file_final[k])
            
            # get the game information
            game_last_name = item_index_game_array[(v_game[-1] - 1)] # the name of the last interacted game in Amazon game
            game_last_id = item_index_mix[game_last_name] # the name of the the last interacted game in Amazon Mix
            game_last_index = np.argwhere(np.array(v_mix_game)==game_last_id)[-1].item()
            user_mix_game = v_mix_game[:game_last_index+1]
            if game_last_id != v_game[-1] + interval:
                print("Wrong!")
            if len(user_mix_game) < 3:
                ipdb.set_trace()
            
            user_train_game_mix[k] = []
            user_train_game_source[k] = []
            user_train_game_target[k] = []
            user_valid_game_target[k] = []
            user_test_game_target[k] = []

            for item in reversed(user_mix_game):
                if item <= interval: # from 1 to 551941, source
                    user_train_game_source[k].append(item)
                    user_train_game_mix[k].append(item)
                elif item >= interval+1: # from 551942 to XXX, target
                    if len(user_test_game_target[k]) == 0:
                        user_test_game_target[k].append(item)
                    elif len(user_valid_game_target[k]) == 0:
                        user_valid_game_target[k].append(item)
                    elif len(user_test_game_target[k]) == 1 and len(user_valid_game_target[k]) == 1:
                        user_train_game_target[k].append(item)
                        user_train_game_mix[k].append(item)

            user_train_game_mix[k].reverse()
            user_train_game_source[k].reverse()
            user_train_game_target[k].reverse()

            pos_mix = len(user_train_game_mix[k])-1
            pos_source = len(user_train_game_source[k])-1
            mix_sequence_for_target_list = []
            source_sequence_for_target_list = []                
            for i in reversed(list(range(0, len(user_train_game_mix[k])))):
                if user_train_game_mix[k][i] <= interval:
                    pos_source = pos_source - 1
                elif user_train_game_mix[k][i] >= interval+1:
                    mix_sequence_for_target_list.append(pos_mix-1)
                    source_sequence_for_target_list.append(pos_source)
                pos_mix = pos_mix - 1
                
                    
            mix_sequence_for_target = mix_sequence_for_target_list[:-1]
            source_sequence_for_target = source_sequence_for_target_list[:-1]
            mix_sequence_for_target.reverse()
            source_sequence_for_target.reverse()

            user_train_game_mix_sequence_for_target[k] = []
            user_train_game_source_sequence_for_target[k] = []
            for x in mix_sequence_for_target:
                user_train_game_mix_sequence_for_target[k].append(x - len(user_train_game_mix[k]))
                    
            for x in source_sequence_for_target:
                user_train_game_source_sequence_for_target[k].append(x - len(user_train_game_source[k]))
                    
        return [user_train_game_mix, user_train_game_source, user_train_game_target, user_valid_game_target, user_test_game_target, user_train_game_mix_sequence_for_target, user_train_game_source_sequence_for_target, usernum, itemnum, interval]    
    
    elif fname == 'amazon_toy':
        user_train_toy_mix = {}
        user_train_toy_source = {}
        user_train_toy_target = {}
        user_valid_toy_target = {}
        user_test_toy_target = {}
        user_train_toy_mix_sequence_for_target = {}
        user_train_toy_source_sequence_for_target = {}
        
        position_mix = []
        position_source = []

        itemnum = len(item_index_mix.keys())
        for k in range(1, len(user_index_overleap.keys()) + 1):
            v_mix_toy = copy.deepcopy(mix_log_file_final[k])
            v_toy = copy.deepcopy(toy_log_file_final[k])

            toy_last_name = item_index_toy_array[(v_toy[-1] - 1)] # the name of the last interacted movie in Amazon Movie
            toy_last_id = item_index_mix[toy_last_name] # the name of the the last interacted movie in Amazon Mix
            toy_last_index = np.argwhere(np.array(v_mix_toy)==toy_last_id)[-1].item()
            user_mix_toy = v_mix_toy[:toy_last_index+1]
            if len(user_mix_toy) < 3:
                ipdb.set_trace()
                
            user_train_toy_mix[k] = []
            user_train_toy_source[k] = []
            user_train_toy_target[k] = []
            user_valid_toy_target[k] = []
            user_test_toy_target[k] = []
            for item in reversed(user_mix_toy):
                if item >= interval+1: # from 551942 to XXX, source
                    user_train_toy_source[k].append(item)
                    user_train_toy_mix[k].append(item)
                elif item <= interval: # from 1 to 551941, target
                    if len(user_test_toy_target[k]) == 0:
                        user_test_toy_target[k].append(item)
                    elif len(user_valid_toy_target[k]) == 0:
                        user_valid_toy_target[k].append(item)
                    elif len(user_test_toy_target[k]) == 1 and len(user_valid_toy_target[k]) == 1:
                        user_train_toy_target[k].append(item)
                        user_train_toy_mix[k].append(item)
            user_train_toy_mix[k].reverse()
            user_train_toy_source[k].reverse()
            user_train_toy_target[k].reverse()
            
            pos_mix = len(user_train_toy_mix[k])-1
            pos_source = len(user_train_toy_source[k])-1
            mix_sequence_for_target_list = []
            source_sequence_for_target_list = []                
            for i in reversed(list(range(0, len(user_train_toy_mix[k])))):
                if user_train_toy_mix[k][i] >= interval+1:
                    pos_source = pos_source - 1
                elif user_train_toy_mix[k][i] <= interval:
                    mix_sequence_for_target_list.append(pos_mix-1)
                    source_sequence_for_target_list.append(pos_source)
                pos_mix = pos_mix - 1
                
            mix_sequence_for_target = mix_sequence_for_target_list[:-1]
            source_sequence_for_target = source_sequence_for_target_list[:-1]
            mix_sequence_for_target.reverse()
            source_sequence_for_target.reverse()
            
            user_train_toy_mix_sequence_for_target[k] = []
            user_train_toy_source_sequence_for_target[k] = []
            for x in mix_sequence_for_target:
                user_train_toy_mix_sequence_for_target[k].append(x - len(user_train_toy_mix[k]))
                    
            for x in source_sequence_for_target:
                user_train_toy_source_sequence_for_target[k].append(x - len(user_train_toy_source[k]))

        return [user_train_toy_mix, user_train_toy_source, user_train_toy_target, user_valid_toy_target, user_test_toy_target, user_train_toy_mix_sequence_for_target, user_train_toy_source_sequence_for_target, usernum, itemnum, interval]   

# evaluate on test set
def evaluate_SASRec(model, dataset, args):
    with torch.no_grad():
        print('Start test...')
        [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval] = dataset

        if args.dataset == 'amazon_toy':
            random_min = 1
            random_max = interval + 1
            item_entries = np.arange(start=random_min, stop=random_max, step=1, dtype=int)
        elif args.dataset == 'amazon_game':
            random_min = interval + 1
            random_max = itemnum + 1
            item_entries = np.arange(start=random_min, stop=random_max, step=1, dtype=int)
        print("The min in source domain is {} and the max in source domain is {}".format(random_min, random_max)) 

        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        NDCG_20 = 0.0
        NDCG_50 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        HT_20 = 0.0
        HT_50 = 0.0
        AUC = 0.0
        loss = 0.0
        valid_user = 0.0
        distance_mix_source = 0
        distance_mix_target = 0
        distance_source_target = 0
        users = range(1, usernum + 1) # range(1, 116255)
        
        labels = torch.zeros(100, device=args.device)
        labels[0] = 1
        for u in users:
            if len(user_train_mix[u]) < 1 or len(user_train_source[u]) < 1 or len(user_train_target[u]) < 1 or len(user_valid_target[u]) < 1 or len(user_test_target[u]) < 1:
                continue

                # for the movie domain
            seq_mix = np.zeros([args.maxlen], dtype=np.int32) # (200,)
            seq_source = np.zeros([args.maxlen], dtype=np.int32) # (200,)
            seq_target = np.zeros([args.maxlen], dtype=np.int32) # (200,)

            idx_mix = args.maxlen - 1 #49
            idx_source = args.maxlen - 1 #49
            idx_target = args.maxlen - 1 #49
            
            for i in reversed(user_train_mix[u]): # reversed是逆序搜索，这里的i指的是交互的物品
                seq_mix[idx_mix] = i
                idx_mix -= 1
                if idx_mix == -1: break

            for i in reversed(user_train_source[u]): # reversed是逆序搜索，这里的i指的是交互的物品
                seq_source[idx_source] = i
                idx_source -= 1
                if idx_source == -1: break

            seq_target[idx_target] = user_valid_target[u][0]
            idx_target -= 1       
            for i in reversed(user_train_target[u]): # reversed是逆序搜索，这里的i指的是交互的物品
                seq_target[idx_target] = i
                idx_target -= 1
                if idx_target == -1: break

            sample_pool = np.setdiff1d(item_entries, seq_target)
            item_idx = np.random.choice(sample_pool, args.num_samples, replace=False)
            item_idx[0] = user_test_target[u][0]

#             ipdb.set_trace()
            predictions, mix_last_feat, source_last_feat, target_last_feat = model.predict(*[np.array(l) for l in [[u], [seq_mix], [seq_source], [seq_target], [item_idx]]])
            
            distance_mix_source += torch.dist(mix_last_feat, source_last_feat, p=2)
            distance_mix_target += torch.dist(mix_last_feat, target_last_feat, p=2)
            distance_source_target += torch.dist(source_last_feat, target_last_feat, p=2)
            
            AUC += roc_auc_score(labels.cpu(), predictions[0].cpu())            

            loss_test = torch.nn.BCEWithLogitsLoss()(predictions[0].detach(), labels)
            loss += loss_test.item()
            predictions = -predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

#             AUC += compute_auc(predictions)
            if rank < 1:
                NDCG_1 += 1 / np.log2(rank + 2)
                HT_1 += 1
            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2)
                HT_5 += 1
            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1
            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1
            if rank < 50:
                NDCG_50 += 1 / np.log2(rank + 2)
                HT_50 += 1
            if valid_user % 1000 == 0:
                print('process test user {}'.format(valid_user))

    return NDCG_1 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_20 / valid_user, NDCG_50 / valid_user, HT_1 / valid_user, HT_5 / valid_user, HT_10 / valid_user, HT_20 / valid_user, HT_50 / valid_user, AUC / valid_user, loss / valid_user, distance_mix_source / valid_user, distance_mix_target / valid_user, distance_source_target / valid_user


