import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel,GPT2LMHeadModel,AutoModel
from util.sampler import next_batch_sequence,next_batch_sequence_for_test
from util.structure import PointWiseFeedForward
from util.loss import l2_reg_loss,InfoNCE,batch_softmax_loss
from util import feature
# from data import pretrain
# from data.pretrain import Pretrain
# from data.sequence import Sequence
import os
import torch.nn.functional as F
from util.algorithm import find_k_largest
from util.metrics import ranking_evaluation
import sys
# import os
# from data.sequence import Sequence
# from util.conf import OptionConf,ModelConf
# from data.loader import FileIO



class Recformer():
    def __init__(self, args, data):
        self.data = data
        self.args = args
        # print("data", data)
        # print("args", args)
        self.bestPerformance = []

        self.max_len = int(self.args.max_lens)
        top = self.args.topK.split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.feature=str(self.args.feature)

        self.block_num = int(2)
        self.drop_rate = float(0.2)
        self.head_num = int(8)
        self.model = Recformer_Model(self.data, self.args.emb_size, self.max_len, self.block_num, self.head_num, self.drop_rate,
                                    self.feature, self.args.data_path, self.args.dataset)
        self.cl_rate = 0.001
        self.rec_loss = torch.nn.BCEWithLogitsLoss()

    def pretrain(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        for epoch in range(self.args.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.args.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, _ = batch
                seq_emb = model.forward(seq, pos)
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss

                # batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', batch_loss.item())
            model.eval()
            self.fast_evaluation(epoch, "pretrain", None)
            # torch.save(model.state_dict(), './model/checkpoint/SASRec_pretrain.pt')

    def save(self, stage):
        if not os.path.exists('./model/'):
                os.makedirs('./model/')
        if not os.path.exists('./model/'+self.args.dataset):
                os.makedirs('./model/'+self.args.dataset)        
        if stage == 'pretrain':
            torch.save(self.model.state_dict(), './model/'+self.args.dataset+'/SASRec_pretrain.pt')
        elif stage == 'finetune':
            torch.save(self.model.state_dict(), './model/'+self.args.dataset+'/SASRec_finetune.pt')

    def finetune(self, attack=None):
        if os.path.exists('./model/'+self.args.dataset+'/SASRec_pretrain.pt'):
            state_dict = torch.load('./model/'+self.args.dataset+'/SASRec_pretrain.pt')
            del state_dict["item_emb"]
            self.model = Recformer_Model(self.data, self.args.emb_size, self.max_len, self.block_num, self.head_num, self.drop_rate,
                                        self.feature, self.args.data_path, self.args.dataset)
            self.model.load_state_dict(state_dict, strict=False)
            print("Loaded the model checkpoint!")

        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lRate)
        for epoch in range(self.args.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_sequence(self.data, self.args.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, seq_len = batch
                y = torch.tensor(y)
                seq_emb = model.forward(seq, pos)
                if (self.feature == 'text'):
                    y_emb = model.mlps(model.bert_tensor[y.cuda()])
                elif(self.feature == 'id'):
                    y_emb = model.item_emb[y]
                elif(self.feature=='id+text'):
                    y_emb = model.item_emb[y]+model.mlps(model.bert_tensor[y.cuda()])
                
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                
                cl_emb1 = [seq_emb[i, 0:last , :].view(-1, self.args.emb_size) for i, last in enumerate(seq_len)]
                cl_emb2 = [y_emb[i, 0:last , :].view(-1, self.args.emb_size) for i, last in enumerate(seq_len)]
                cl_loss_item = self.cl_rate * InfoNCE(torch.cat(cl_emb1, 0), torch.cat(cl_emb2, 0), 1)
                
                batch_loss = rec_loss + cl_loss_item
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', batch_loss.item())
            model.eval()
            self.fast_evaluation(epoch, "finetune", attack)
        
    def calculate_loss(self, seq_emb, y, neg,pos):
        y = torch.tensor(y)
        neg = torch.tensor(neg)
        if (self.feature == 'text'):
            outputs = self.model.mlps(self.model.bert_tensor[y.cuda()])
            y_emb=outputs
            outputs = self.model.mlps(self.model.bert_tensor[neg.cuda()])
            neg_emb=outputs

        elif(self.feature == 'id'):
            y_emb = self.model.item_emb[y]
            neg_emb = self.model.item_emb[neg]
        elif(self.feature=='id+text'):
            y_emb = self.model.item_emb[y]+self.model.mlps(self.model.bert_tensor[y.cuda()])
            neg_emb = self.model.item_emb[neg]+self.model.mlps(self.model.bert_tensor[neg.cuda()])
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
        indices = np.where(pos != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        for n, batch in enumerate(next_batch_sequence_for_test(self.data, self.args.batch_size,max_len=self.max_len)):
            seq, pos, seq_len = batch
            seq_names = [seq_full[0] for seq_full in self.data.original_seq[n*self.args.batch_size:(n+1)*self.args.batch_size]]

            candidates = self.predict(seq, pos, seq_len)
            for name,res in zip(seq_names,candidates):
                ids, scores = find_k_largest(self.max_N, res)
                item_names = [self.data.id2item[iid] for iid in ids if iid!=0 and iid<=self.data.item_num]
                rec_list[name] = list(zip(item_names, scores))
            if n % 100 == 0:
                process_bar(n, self.data.raw_seq_num/self.args.batch_size)
        process_bar(self.data.raw_seq_num, self.data.raw_seq_num)
        print('')
        return rec_list, ranking_evaluation(self.data.test_set, rec_list, [self.max_N])

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.args.emb_size) for i,last in enumerate(seq_len)]
            if self.feature == 'text' or self.feature=='id+text':
                item_feature_emb = self.model.mlps(self.model.bert_tensor)

            if self.feature == 'text':
                score = torch.matmul(torch.cat(last_item_embeddings,0), item_feature_emb.transpose(0, 1))
            elif self.feature=='id':
                score = torch.matmul(torch.cat(last_item_embeddings,0), self.model.item_emb.transpose(0, 1))
            elif self.feature=='id+text':
                score = torch.matmul(torch.cat(last_item_embeddings,0), (self.model.item_emb+item_feature_emb).transpose(0, 1))
        return score.cpu().numpy()

    def fast_evaluation(self, epoch, stage, attack):
        print('Evaluating the model...')
        rec_list, _ = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                print(epoch, 'saved')
                if attack is None:
                    print("attack1", attack)
                    self.save(stage)
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            print(epoch," saved")
            if attack is None:
                print("attack1", attack)
            #     self.save(stage)
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        # bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + ' | '
        #bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        # if epoch==self.args.maxEpoch-1:
        print('*Best Performance* ')
        print('Epoch:'+str(self.bestPerformance[0]) + ','+ bp)
        print('-' * 120)
        return measure




class Recformer_Model(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, data_path, dataset):
        super(Recformer_Model, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.data_path = data_path
        self.dataset = dataset
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_

        if (self.feature == 'text' or self.feature == 'id+text'):
            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
            self.bert_tensor_kv = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
            if (len(self.dataset.split(",")) == 1):
                # if not os.path.exists(self.data_path + self.dataset + "/whole_tensor.pt"):
                mask = 0
                pre = Pretrain(self.data, self.data_path, self.dataset, mask)
                self.token_input, self.token_mask, kv_whole_tensor =  pre.train_inputs, pre.train_masks, pre.kv_whole_tensor
                tensor = torch.load(self.data_path+ self.dataset + "/whole_tensor.pt")
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor_kv = torch.cat([self.bert_tensor_kv, kv_whole_tensor], 0)
                self.mlps = MLPS(self.emb_size)
            elif (len(self.dataset.split(",")) > 1):
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.dataset.split(","):
                    if not os.path.exists(self.data_path+ self.dataset + "/whole_tensor.pt"):
                        mask = 0
                        pre = Pretrain(self.data, self.data_path, self.dataset, mask)
                    tensor = torch.load(self.data_path+ self.dataset + "/whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.mlps = MLPS(self.emb_size)

        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 1, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len + 1, self.emb_size)))
        self.token_pos_emb = nn.Parameter(initializer(torch.empty(self.token_input.shape[0] + 2, self.emb_size)))
        # print(self.token_pos_emb.shape)
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer = torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))

            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, seq, pos):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        if (self.feature == 'text'):
            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
            kv_emb = self.mlps(self.bert_tensor_kv[seq.cuda()])
            seq_emb = seq_emb * self.emb_size ** 0.5
            kv_emb = kv_emb * self.emb_size ** 0.5
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
            seq_emb = seq_emb * self.emb_size ** 0.5
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq.cuda()])
            kv_emb = self.mlps(self.bert_tensor_kv[seq.cuda()])
            seq_emb = seq_emb * self.emb_size ** 0.5
            kv_emb = kv_emb * self.emb_size ** 0.5
        # print("seq_emb", seq_emb.shape)

        # print("pos", len(pos), len(pos[0]))
        # print("self.pos_emb", len(self.pos_emb),len(self.pos_emb[0]))
        # print("seq_emb", seq_emb.shape)
        # print("seq", seq)
        token_input_id = self.token_mask[seq-1]
        # print("token_input_id", token_input_id.shape)
        # token_input_id = torch.sum(token_input_id, dim=2)
        # print("token_input_id", token_input_id.shape)
        token_pos_emb = self.token_pos_emb[token_input_id.cuda()]
        token_pos_emb = torch.sum(token_pos_emb, dim=2)
        # print("token_pos_emb", token_pos_emb.shape)
        pos_emb = self.pos_emb[pos]
        # seq_emb = seq_emb + pos_emb + token_pos_emb
        if self.feature == 'id':
            seq_emb = seq_emb + pos_emb
        else:
            seq_emb = seq_emb + pos_emb + kv_emb + token_pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb


# encoder
class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()
        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.H),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):
        logits = self.classifier(bert_tensor)
        return logits


# Paper: Self-Attentive Sequential Recommendation

class Pretrain(object):
    def __init__(self, data, data_path, dataset, mask):
        # self.datasetfile=datasetfile
        self.data=data
        self.bert=Bert().cuda()
        initializer = nn.init.xavier_uniform_
        self.dataset = dataset
        self.train_inputs, self.train_masks=eval('read_feature'+'(\''+ data_path + '\',\'' + dataset+'\', self.data.id2item'+')')
        self.train_inputs_kv, self.train_masks_kv=eval('read_kv_feature'+'(\''+ data_path + '\',\'' + dataset+'\', self.data.id2item'+')')
        
        # print("111", self.train_inputs.shape)
        # whole_list = []
        batch_size = 32
        whole_tensor = None
        self.kv_whole_tensor = None
        for i in range(0, len(self.train_inputs), batch_size):
            batch_inputs = self.train_inputs[i:i+batch_size].cuda()
            batch_masks = self.train_masks[i:i+batch_size].cuda()
            # batch_inputs_kv = self.train_inputs_kv[i:i+batch_size].cuda()
            # batch_masks_kv = self.train_masks_kv[i:i+batch_size].cuda()
            with torch.no_grad():  # 减少内存使用
                outputs = self.bert(batch_inputs, batch_masks)[0][:, 0, :]
                # outputs_kv = self.bert(batch_inputs_kv, batch_masks_kv)[0][:, 0, :]

            if whole_tensor is None:
                whole_tensor = outputs
            else:
                whole_tensor = torch.cat([whole_tensor, outputs], 0)
                
            # if self.kv_whole_tensor is None:
            #     self.kv_whole_tensor = outputs_kv
            # else:
            #     self.kv_whole_tensor = torch.cat([self.kv_whole_tensor, outputs_kv], 0)
            # print(whole_tensor.shape)
            # 释放GPU内存
            # del batch_inputs, batch_masks, outputs, batch_inputs_kv, batch_masks_kv, outputs_kv
            del batch_inputs, batch_masks, outputs
            torch.cuda.empty_cache()

        torch.save(whole_tensor, data_path+dataset+"/whole_tensor.pt")
        
        for i in range(0, len(self.train_inputs), batch_size):
            batch_inputs_kv = self.train_inputs_kv[i:i+batch_size].cuda()
            batch_masks_kv = self.train_masks_kv[i:i+batch_size].cuda()
            with torch.no_grad():  # 减少内存使用
                outputs_kv = self.bert(batch_inputs_kv, batch_masks_kv)[0][:, 0, :]
                
            if self.kv_whole_tensor is None:
                self.kv_whole_tensor = outputs_kv
            else:
                self.kv_whole_tensor = torch.cat([self.kv_whole_tensor, outputs_kv], 0)
            # print(whole_tensor.shape)
            # 释放GPU内存
            del batch_inputs_kv, batch_masks_kv, outputs_kv
            torch.cuda.empty_cache()
        
        



# class Pretrain(object):
#     def __init__(self, data, data_path, dataset, mask):
#         # self.datasetfile=datasetfile
#         self.data=data
#         self.bert=Bert().cuda()
#         initializer = nn.init.xavier_uniform_
#         self.dataset = dataset
#         self.train_inputs, self.train_masks=eval('read_feature'+'(\''+ data_path + '\',\'' + dataset+'\', self.data.id2item'+')')
        
#         self.train_inputs_kv, self.train_masks_kv=eval('read_kv_feature'+'(\''+ data_path + '\',\'' + dataset+'\', self.data.id2item'+')')
        
        
#         # print("111", self.train_inputs.shape)
#         whole_list = []
#         i = 0
#         while len(self.train_inputs) > ((i+1) * 100):
#             outputs = self.bert(self.train_inputs[i*100:(i+1)*100].cuda(), self.train_masks[i*100:(i+1)*100].cuda())[0][:, 0, :]
#             whole_list.append(outputs)
#             i = i + 1

#         outputs = self.bert(self.train_inputs[i*100:len(self.train_inputs)].cuda(), self.train_masks[i*100:len(self.train_inputs)].cuda())[0][:, 0, :]

#         whole_list.append(outputs)
#         whole_tensor = whole_list[0]
#         for i in range(1, len(whole_list)):
#             whole_tensor = torch.cat([whole_tensor, whole_list[i]], 0)

#         # self.train_masks = torch.cat([torch.tensor([self.train_masks[0]]), self.train_masks], 0)
        
#         torch.save(whole_tensor, data_path+dataset+"/whole_tensor.pt")
        
#         whole_list = []
#         i = 0
#         while len(self.train_inputs_kv) > ((i+1) * 100):
#             outputs = self.bert(self.train_inputs_kv[i*100:(i+1)*100].cuda(), self.train_masks_kv[i*100:(i+1)*100].cuda())[0][:, 0, :]
#             whole_list.append(outputs)
#             i = i + 1

#         outputs = self.bert(self.train_inputs_kv[i*100:len(self.train_inputs_kv)].cuda(), self.train_masks_kv[i*100:len(self.train_inputs_kv)].cuda())[0][:, 0, :]

#         whole_list.append(outputs)
#         self.kv_whole_tensor = whole_list[0]
#         for i in range(1, len(whole_list)):
#             self.kv_whole_tensor = torch.cat([self.kv_whole_tensor, whole_list[i]], 0)


        

    def execute(self):
        pass
        
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        # self.bert = BertModel.from_pretrained('bert')
        self.bert = AutoModel.from_pretrained('longformer')
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        return outputs





import csv
import re
import pandas as pd
import numpy as np
from re import split
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, GPT2Tokenizer
from transformers import BertModel,GPT2LMHeadModel
from transformers import LongformerTokenizer
import nltk
from nltk.corpus import stopwords
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import json
import time

max_len = 150
# tokenizer = BertTokenizer.from_pretrained('bert', do_lower_case=True)
tokenizer = LongformerTokenizer.from_pretrained('longformer', do_lower_case=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def text_preprocessing(s):
    s = s.lower()

    s = re.sub(r"\'t", " not", s)

    s = re.sub(r'(@.*?)[\s]', ' ', s)

    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)

    s = re.sub(r'([\;\:\|•«\n])', ' ', s)

    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])

    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),
            add_special_tokens=True,
            max_length=max_len ,
            pad_to_max_length=True,
            # return_tensors='pt',
            return_attention_mask=True ,
        )
        # Add the outputs to the lists
        
        # if(len(input_ids)==0):
        #     input_ids.append(encoded_sent.get('input_ids'))
        # if(len(attention_masks)==0):
        #     attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # [total sentence, 100]
    return input_ids, attention_masks

def read_feature(data_path, dataset, id2item):
    featureOriginal = {}
    featureNew = []
    with open(data_path+dataset+"/feature.txt", 'r', encoding="gbk",
              errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            item = line.split(":")
            featureOriginal[item[0].strip()]=item[1].strip()
    for i in id2item:
        try:
            # print(i, id2item[i], featureOriginal[id2item[i]])
            featureNew.append(featureOriginal[id2item[i]])
        except:
            print('None')
            featureNew.append('None')
        # print(featureOriginal[split('-',id2item[i])[0]])
        # print(featureNew)
    train_inputs, train_masks = preprocessing_for_bert(featureNew)
    return train_inputs, train_masks


def text_to_dict(text):
    title_pos = text.find("title")
    brand_pos = text.find("brand")
    description_pos = text.find("description")

    title_content = text[title_pos + len("title "):brand_pos].strip()
    brand_content = text[brand_pos + len("brand "):description_pos].strip()
    description_content = text[description_pos + len("description "):].strip()

    result_dict = {
        "title": title_content,
        "brand": brand_content,
        "description": description_content
    }

    return result_dict


def preprocessing_for_bert_kv(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        dict_sent = text_to_dict(sent)
        for key, value in dict_sent.items():
            one_input_ids = []
            one_attention_masks = []
            if key == 'title':
                encoded_sent = tokenizer.encode_plus(
                    text=text_preprocessing(key+value),
                    add_special_tokens=True,
                    max_length=50 ,
                    pad_to_max_length=True,
                    # return_tensors='pt',
                    return_attention_mask=True ,
                )
            if key == 'brand':
                encoded_sent = tokenizer.encode_plus(
                    text=text_preprocessing(key+value),
                    add_special_tokens=True,
                    max_length=10 ,
                    pad_to_max_length=True,
                    # return_tensors='pt',
                    return_attention_mask=True ,
                )  
            if key == 'description':
                encoded_sent = tokenizer.encode_plus(
                    text=text_preprocessing(key+value),
                    add_special_tokens=True,
                    max_length=90 ,
                    pad_to_max_length=True,
                    # return_tensors='pt',
                    return_attention_mask=True ,
                )      
            one_input_ids.append(encoded_sent.get('input_ids'))
            one_attention_masks.append(encoded_sent.get('attention_mask'))
            # print("11", one_input_ids, one_attention_masks)
        one_input_ids_flatten = sum(one_input_ids,[]) 
        one_attention_masks_flatten = sum(one_attention_masks,[])
        # print("22", one_input_ids_flatten, one_attention_masks_flatten)    
        input_ids.append(one_input_ids_flatten)
        attention_masks.append(one_attention_masks_flatten)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    print(input_ids)
    print(attention_masks)
    # [total sentence, 100]
    return input_ids, attention_masks


def read_kv_feature(data_path, dataset, id2item):
    featureOriginal = {}
    featureNew = []
    with open(data_path+dataset+"/feature.txt", 'r', encoding="gbk",
              errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            item = line.split(":")
            featureOriginal[item[0].strip()]=item[1].strip()
    for i in id2item:
        try:
            # print(i, id2item[i], featureOriginal[id2item[i]])
            featureNew.append(featureOriginal[id2item[i]])
        except:
            print('None')
            featureNew.append('None')
        # print(featureOriginal[split('-',id2item[i])[0]])
        # print(featureNew)
    train_inputs, train_masks = preprocessing_for_bert_kv(featureNew)
    return train_inputs, train_masks

