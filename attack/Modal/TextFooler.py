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
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')



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


    
class TextFooler():
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
        
        max_len = 150
        self.tokenizer = BertTokenizer.from_pretrained('bert', do_lower_case=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def posionDataAttack(self, recommender):

        
        alternative_features = self.generatepopularfeature(len(self.targetItem))
        self.textfooler(alternative_features, recommender)
        # for targetitem_i in range(len(self.targetItem)):
        #     self.data.feature_data = self.textbugger(alternative_features, recommender)
            # self.data.feature_data[self.targetItem[targetitem_i]] = alternative_features[targetitem_i]

        return self.data.training_data, self.data.feature_data

        # return vstack([self.interact, fakeRat])

    def textfooler(self, features, recommender):
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
            # print(tmptargetfeature)
            feature_token = feature.split(" ")
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
            # attack_loss = []
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
            importance_feature_token = feature_token[largeindex-1]
            # print("importance_feature_token", importance_feature_token)
            importance_feature_embedding = self.get_embedding(importance_feature_token)
            
            cos = torch.nn.CosineSimilarity(dim=1)
            vocab = self.tokenizer.get_vocab()
            similarity = {}

            for word, index in vocab.items():
                # print("word", word)
                token_embedding = self.get_embedding(word)
                # print("importance_feature_embedding", importance_feature_embedding.shape, token_embedding.shape)
                cos_sim = cos(importance_feature_embedding, token_embedding)
                similarity[word] = cos_sim.item()
            # print("similarity", similarity)
            # Sorting and getting top 50 tokens
            top_50_tokens = sorted(similarity, key=similarity.get, reverse=True)[:50]

            # print("top_50_tokens", top_50_tokens)
            
            importance_feature_token_pos = nltk_pos_tag(importance_feature_token)

            # Filter tokens with the same POS
            filtered_tokens = [token for token in top_50_tokens if nltk_pos_tag(token) == importance_feature_token_pos]

            # print("filtered_tokens", filtered_tokens)
            
            bugfeature = []
            for token in filtered_tokens:
                bugfeature.append(tmptargetfeature.replace(importance_feature_token, token))
            # print("bugfeature", bugfeature)
            
            
            
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
            
            self.data.feature_data[self.targetItem[i]] = None
            flaglargeindex = int(torch.argmax(score_list))
            while self.data.feature_data[self.targetItem[i]]  == None:
                tmplargeindex = int(torch.argmax(score_list))
                tmptoken = token_outputs[tmplargeindex]
                
                if F.cosine_similarity(target_token_outputs[0], tmptoken, dim=0) > 0.8:
                    self.data.feature_data[self.targetItem[i]] = bugfeature[tmplargeindex]
                if score_list[tmplargeindex] == 0:
                    # print("Attention: low similarity!!!")
                    # print("score_list", score_list)
                    self.data.feature_data[self.targetItem[i]] = bugfeature[tmplargeindex]
                    break
                score_list[tmplargeindex] = 0
            
    def get_embedding(self, token):
        input_ids = []
        attention_masks = []
  
        encoded_sent = tokenizer.encode_plus(
            text=token,
            add_special_tokens=True,
            max_length=15,
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
        # print("input_ids", input_ids.shape)
        # print("inputs", inputs)
        with torch.no_grad():
            # print("1111", self.bert(input_ids.cuda(), attention_masks.cuda())[0].shape)
            outputs = self.bert(input_ids.cuda(), attention_masks.cuda())[0][:, 0, :]
        return outputs
                        

        
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



# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Function to get NLTK POS tag
def nltk_pos_tag(token):
    tag = nltk.pos_tag([token])[0][1]
    return get_wordnet_pos(tag)

