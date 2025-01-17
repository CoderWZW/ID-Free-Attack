import numpy as np
import random
from util.tool import targetItemGenerateModal, getModalpopularItem
from util.sampler import next_batch_sequence,next_batch_sequence_for_test
from scipy.sparse import vstack, csr_matrix
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel,GPT2LMHeadModel
from transformers import BertTokenizer, GPT2Tokenizer
from copy import deepcopy
import torch
import jieba
import jieba.analyse
import jieba.posseg as pseg
import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')

class TextBugger():
    def __init__(self, arg, data):
        self.data = data
        self.targetItem, self.targetfeature = targetItemGenerateModal(data, arg)
        print(len(self.targetItem), len(self.targetfeature), self.targetItem, self.targetfeature)
        self.targetItem_id = [data.item[item_key.replace('\'', '').strip()] for item_key in self.targetItem]
        self.itemNum = data.item_num

        # # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = int(data.averagefeedback)
        # print("maliciousFeedbackSize", self.maliciousFeedbackSize)
        # if self.maliciousFeedbackSize == 0:
        #     self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        # elif self.maliciousFeedbackSize >= 1:
        #     self.maliciousFeedbackNum = self.maliciousFeedbackSize
        # else:
        #     self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)
        
        self.bert=Bert().cuda()

    def posionDataAttack(self, recommender):

        # alternative_features = self.generatepopularfeature(len(self.targetItem))
        alternative_features = []
        for target_item in self.targetItem:
            alternative_features.append(self.data.feature_data[target_item])
        self.textbugger(alternative_features, recommender)
        # for targetitem_i in range(len(self.targetItem)):
        #     self.data.feature_data = self.textbugger(alternative_features, recommender)
            # self.data.feature_data[self.targetItem[targetitem_i]] = alternative_features[targetitem_i]

        return self.data.training_data, self.data.feature_data

        # return vstack([self.interact, fakeRat])

    def textbugger(self, features, recommender):
        tmpRecommender = recommender
        
        model = tmpRecommender.model.cuda()
        
        target_input_ids, target_attention_masks = preprocessing_for_bert(features)

        target_token_outputs = self.bert(target_input_ids.cuda(), target_attention_masks.cuda())[0][:, 0, :]
        # alternative_features = self.generatepopularfeature(len(self.targetItem))
        # for targetitem_i in range(len(self.targetItem)):
        #     # print(targetitem_i)
        #     self.data.feature_data[self.targetItem[targetitem_i]] = alternative_features[targetitem_i]
        for i in range(len(features)):  
            # find import words
            feature = features[i]
            tmptargetfeature = feature
            feature_token = tmptargetfeature.split(" ")
            deletedfeature = [tmptargetfeature]
            for token in feature_token:
                deletedfeature.append(tmptargetfeature.replace(token, ""))
                
            # for line in deletedfeature:
            #     print("line", line)
                            
            # words = feature.split(" ")
            # print("words", words, len(words))
            importance_input_ids, importance_attention_masks = preprocessing_for_bert(deletedfeature)
            # print("input_ids", input_ids.shape)
            # calculate the item_feature_embedding
            
            # 改变后的样子
            importance_token_outputs = self.bert(importance_input_ids.cuda(), importance_attention_masks.cuda())[0][:, 0, :]
            token_embs = model.mlps(importance_token_outputs)
            score_list = []
            for token_emb in token_embs:
                score = 0
                for n, batch in enumerate(next_batch_sequence(self.data, 32, max_len=50)):
                    seq, pos, y, neg_idx, seq_len = batch
                    with torch.no_grad():
                        seq_emb = model.forward(seq,pos)
                        last_item_embeddings = [seq_emb[i,last-1,:].view(-1,64) for i,last in enumerate(seq_len)]
                        score += torch.mul(torch.cat(last_item_embeddings,0), token_emb).sum()
                    del seq_emb
                    torch.cuda.empty_cache()
                score_list.append(score)
            score_list = torch.tensor(score_list)
            
            # print("score_list1", score_list)
            score_list = np.abs(score_list - score_list[0])
            # print("score_list2", score_list)

            largeindex = int(torch.argmax(score_list))
            import_word = feature_token[largeindex-1]
            # print("import_word", import_word)

            # print("feature", features)
            # keywords = jieba.analyse.extract_tags(feature, topK=30)
            # print("keywords", keywords)
            
            # tagged_keywords = nltk.pos_tag(keywords)

            # adjectives_in_keywords = [word for word, tag in tagged_keywords if tag in ('JJ', 'JJR', 'JJS')]

            # print("adjectives", adjectives_in_keywords)
            
            # if len(adjectives_in_keywords) == 0:
            #     import_word = keywords[0]
            # else:
            #     import_word = adjectives_in_keywords[0]
            # print("import_word", import_word)
            
            bugs = self.generateBugs(import_word)

            # print(bugs)
            
            tmptargetfeature = feature
            bugfeature = []
            for key, bug in bugs.items():
                bugfeature.append(tmptargetfeature.replace(import_word, bug))
            # print("bugfeature", len(bugfeature), bugfeature)
                            
            # words = feature.split(" ")
            # print("words", words, len(words))
            input_ids, attention_masks = preprocessing_for_bert(bugfeature)
            # print("input_ids", input_ids.shape)
            # calculate the item_feature_embedding
            
            # 改变后的样子
            token_outputs = self.bert(input_ids.cuda(), attention_masks.cuda())[0][:, 0, :]
            token_embs = model.mlps(token_outputs)
            score_list = []
            for token_emb in token_embs:
                score = 0
            # attack_loss = []
                for n, batch in enumerate(next_batch_sequence(self.data, 256, max_len=50)):
                    seq, pos, y, neg_idx, seq_len = batch
                    with torch.no_grad():
                        seq_emb = model.forward(seq,pos)
                        last_item_embeddings = [seq_emb[i,last-1,:].view(-1,64) for i,last in enumerate(seq_len)]
                        score += torch.mul(torch.cat(last_item_embeddings,0), token_emb).sum()
                    del seq_emb
                    torch.cuda.empty_cache()
                score_list.append(score)
                # print("score_list", score_list)
            score_list = torch.tensor(score_list)
            # print(score_list)
            self.data.feature_data[self.targetItem[i]] = None
            flaglargeindex = int(torch.argmax(score_list))
            while self.data.feature_data[self.targetItem[i]]  == None:
                tmplargeindex = int(torch.argmax(score_list))
                tmptoken = token_outputs[tmplargeindex]
                
                # print(tmplargeindex)
                # print(score_list)
                # print(F.cosine_similarity(target_token_outputs[0], tmptoken, dim=0))
                if F.cosine_similarity(target_token_outputs[0], tmptoken, dim=0) > 0.8:
                    self.data.feature_data[self.targetItem[i]] = bugfeature[tmplargeindex]
                    
                if score_list[tmplargeindex] == 0:
                    # print("Attention: low similarity!!!")
                    # print("score_list", score_list)
                    self.data.feature_data[self.targetItem[i]] = bugfeature[tmplargeindex]
                    break
                score_list[tmplargeindex] = 0
                    # print(self.data.feature_data[self.targetItem[i]])
            # print("self.data.feature_data[self.targetItem[i]]", self.data.feature_data[self.targetItem[i]])
        # print(self.data.feature_data)
        # print("self.data.feature_data", self.data.feature_data)       
                
                
    def generateBugs(self, word, sub_w_enabled=False, typo_enabled=False):
        bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}
        if len(word) <= 2:
            return bugs
        bugs["insert"] = bug_insert(word)
        bugs["delete"] = bug_delete(word)
        bugs["swap"] = bug_swap(word)
        bugs["sub_C"] = bug_sub_C(word)
        # bugs["sub_W"] = bug_sub_W(word)
        return bugs            
        
    def generatepopularfeature(self, target_item_num):
        popular_item = getModalpopularItem(self.data)
        select_popular_item = popular_item[-target_item_num:]
        # select_popular_item = random.sample(popular_item, target_item_num)
        # print("popular_item", select_popular_item)
                    
        sample_text=[]
        for item_key in select_popular_item:
            if item_key in self.data.feature_data:
                sample_text.append(self.data.feature_data[item_key])    
            # sample_text = random.sample(self.data.feature_data[popular_item],target_item_num)
        return sample_text

import csv
import re
import pandas as pd
import numpy as np
from re import split
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, GPT2Tokenizer
from transformers import BertModel,GPT2LMHeadModel
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
tokenizer = BertTokenizer.from_pretrained('bert', do_lower_case=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert')
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        return outputs

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

# from ...exceptions import WordNotInDictionaryException

# def bug_sub_W(word):
#     try:
#         res = self.substitute(word, None)
#         if len(res) == 0:
#             return word
#         return res[0][0]
#     except:
#         return word

def bug_insert(word):
    if len(word) >= 6:
        return word
    res = word
    point = random.randint(1, len(word) - 1)
    res = res[0:point] + " " + res[point:]
    return res

def bug_delete(word):
    res = word
    point = random.randint(1, len(word) - 2)
    res = res[0:point] + res[point + 1:]
    return res

def bug_swap(word):
    if len(word) <= 4:
        return word
    res = word
    points = random.sample(range(1, len(word) - 1), 2)
    a = points[0]
    b = points[1]

    res = list(res)
    w = res[a]
    res[a] = res[b]
    res[b] = w
    res = ''.join(res)
    return res

def bug_sub_C(word):
    res = word
    key_neighbors = get_key_neighbors()
    point = random.randint(0, len(word) - 1)

    if word[point] not in key_neighbors:
        return word
    choices = key_neighbors[word[point]]
    subbed_choice = choices[random.randint(0, len(choices) - 1)]
    res = list(res)
    res[point] = subbed_choice
    res = ''.join(res)

    return res

def get_key_neighbors():
    ## TODO: support other language here
    # By keyboard proximity
    neighbors = {
        "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
        "i": "uojkl", "o": "ipkl", "p": "ol",
        "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
        "j": "yuihknm", "k": "uiojlm", "l": "opk",
        "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
    }
    # By visual proximity
    neighbors['i'] += '1'
    neighbors['l'] += '1'
    neighbors['z'] += '2'
    neighbors['e'] += '3'
    neighbors['a'] += '4'
    neighbors['s'] += '5'
    neighbors['g'] += '6'
    neighbors['b'] += '8'
    neighbors['g'] += '9'
    neighbors['q'] += '9'
    neighbors['o'] += '0'

    return neighbors