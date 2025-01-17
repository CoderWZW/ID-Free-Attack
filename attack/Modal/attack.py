import numpy as np
import random
from util.tool import targetItemGenerateModal, getModalpopularItem, divideIteminPopularity, ModaldataSave, ModalfeatureSave
from scipy.sparse import vstack, csr_matrix
from util.sampler import next_batch_sequence,next_batch_sequence_for_test

import csv
import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig
# import spacy
# from spacy import displacy
from gensim.summarization import keywords
from util.metrics import AttackMetric,ModalAttackMetric

import warnings
warnings.filterwarnings('ignore')

import asyncio
import time
from collections import defaultdict
from zhipuai import ZhipuAI
import re

class AgentAttack():
    def __init__(self, arg, data):
        self.data = data
        self.targetItem, self.targetfeature = targetItemGenerateModal(data, arg)
        print(len(self.targetItem), len(self.targetfeature), self.targetItem, self.targetfeature)
        self.targetItem_id = [data.item[item_key.replace('\'', '').strip()] for item_key in self.targetItem]
        self.itemNum = data.item_num

        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = int(data.averagefeedback)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)
        
        self.bert=Bert().cuda()

        self.TOKENIZER, self.MODEL = load_model()
        
        self.recommend_item_topN = 10
        self.agent = Agent()

        # self.nlp = spacy.load('en_core_web_sm')

    def initialposionDataAttack(self, recommender):        
        alternative_features = self.initialfeature(self.targetItem, recommender)
        # print("alternative_features", alternative_features)
        print("Generated text:", alternative_features[0])
        for targetitem_i in range(len(self.targetItem)):
            # print(targetitem_i)
            self.data.feature_data[self.targetItem[targetitem_i]] = alternative_features[targetitem_i]
        return self.data.training_data, self.data.feature_data

    def initialfeature(self, targetItem, recommender):
        for targetitem_i in range(len(self.targetItem)):
            target_item_text = self.data.feature_data[targetItem[targetitem_i]]
            print("Original text:", target_item_text)

        popular_important_word = self.get_popular_improtant_word()
        print("popular_important_word", popular_important_word)
        
        most_popular_item = getmostpopularitem(self.data)
        most_popular_item_str = self.data.feature_data[most_popular_item[0]]
        print("most_popular_item_str", most_popular_item_str)
        
        tmpRecommender = recommender
        recommend_important_word = self.get_recommend_important_word(tmpRecommender)
        print("recommend_important_word", recommend_important_word)
        # model = tmpRecommender.model.cuda()
        
        target_item_num = len(targetItem)
        SENTENCES = target_item_num
        num_generated = 1
        result = []
        
        # original_content = "<original content>: " + target_item_text + "\n"
        target_content = most_popular_item_str
        # importantword = "<important words>: " + str(popular_important_word + recommend_important_word) + "\n"
        importantword = str(popular_important_word)
        
        # hitratio, precision, recall, ndcg = self.attack_test(tmpRecommender)
        # print(f"Metrics - Hit Ratio: {hitratio}, Precision: {precision}, Recall: {recall}, NDCG: {ndcg}")
        
        asyncio.run(initialization(self.agent, target_content, importantword))
        # asyncio.run(summarize_history(self.agent))

        print("rewritten content:", self.agent.chat_history_rewritten_content[-1])
        print("reason:", self.agent.chat_history_reason[-1])

        result.append(self.agent.chat_history_rewritten_content[-1])

        return result


    def posionDataAttack(self, recommender, attack_metric):        
        alternative_features = self.generatepopularfeature(self.targetItem, recommender, attack_metric)
        # print("alternative_features", alternative_features)
        print("Generated text:", alternative_features[0])
        for targetitem_i in range(len(self.targetItem)):
            # print(targetitem_i)
            self.data.feature_data[self.targetItem[targetitem_i]] = alternative_features[targetitem_i]
        return self.data.training_data, self.data.feature_data

    def generatepopularfeature(self, targetItem, recommender, attack_metric):

        for targetitem_i in range(len(self.targetItem)):
            target_item_text = self.data.feature_data[targetItem[targetitem_i]]
            print("Original text:", target_item_text)

        popular_important_word = self.get_popular_improtant_word()
        print("popular_important_word", popular_important_word)
        
        most_popular_item = getmostpopularitem(self.data)
        most_popular_item_str = self.data.feature_data[most_popular_item[0]]
        print("most_popular_item_str", most_popular_item_str)
        
        tmpRecommender = recommender
        recommend_important_word = self.get_recommend_important_word(tmpRecommender)
        print("recommend_important_word", recommend_important_word)
        # model = tmpRecommender.model.cuda()
        
        target_item_num = len(targetItem)
        SENTENCES = target_item_num
        num_generated = 1
        result = []
        
        # original_content = "<original content>: " + target_item_text + "\n"
        target_content = most_popular_item_str
        # importantword = "<important words>: " + str(popular_important_word + recommend_important_word) + "\n"
        importantword = str(popular_important_word)

        asyncio.run(discussion(self.agent, target_content, importantword, attack_metric))

        print(f"Generated title: {self.agent.chat_history_rewritten_content[-1]}")
            
        result.append(self.agent.chat_history_rewritten_content[-1])

        return result

    def attack_test(self,recommender):
        rec_list, self.attackRecommendresult = recommender.test()
        attackmetrics = ModalAttackMetric(rec_list, self.targetItem)
        return attackmetrics.hitRate(),attackmetrics.precision(),attackmetrics.recall(),attackmetrics.NDCG()

    def get_popular_improtant_word(self):
        popular_item_str_list = []
        popular_item, unpopular_item = divideIteminMostPopularity(self.data)
        # self.data.feature_data[popular_item[0]]

        for item in popular_item:
            if item in self.data.feature_data:
                if check_length(self.data.feature_data[item]):
                    popular_item_str = self.data.feature_data[item]
                    popular_item_str_list.append(popular_item_str)
        
        all_popular_items_description = ' '.join(popular_item_str_list)

        extracted_keywords = keywords(all_popular_items_description, words=20, lemmatize=True).split('\n')
        base_words = ['title', 'description', 'skin']
        for base_word in base_words:
            if base_word in extracted_keywords:
                extracted_keywords.remove(base_word)
        return extracted_keywords

    def get_recommend_important_word(self, recommender):
        rec_list, _ = recommender.test()
        # print("rec_list", rec_list)
        recommend_item_frequency = {}
        u_num = 0
        for user in rec_list:
            u_num += 1
            for item in rec_list[user]:
                item_name = item[0]
                if item_name in recommend_item_frequency:
                    recommend_item_frequency[item_name] += 1
                else:
                    recommend_item_frequency[item_name] = 1
            if u_num > 50:
                break
        # print("recommend_item_frequency", recommend_item_frequency)
        sorted_dict_recommend_item_frequency = dict(sorted(recommend_item_frequency.items(), key=lambda item: item[1]))
        selected_keys = list(sorted_dict_recommend_item_frequency.keys())[-self.recommend_item_topN:]

        popular_item_str_list = []
        for item in selected_keys:
            if item in self.data.feature_data:
                if check_length(self.data.feature_data[item]):
                    popular_item_str = self.data.feature_data[item]
                    popular_item_str_list.append(popular_item_str)
        
        # print("popular_item_str_list", popular_item_str_list)
        all_popular_items_description = ' '.join(popular_item_str_list)

        extracted_keywords = keywords(all_popular_items_description, words=20, lemmatize=True).split('\n')
        base_words = ['title', 'description', 'skin']
        for base_word in base_words:
            if base_word in extracted_keywords:
                extracted_keywords.remove(base_word)
        return extracted_keywords
    
    def get_embedding(self, token, tokenizer):
        input_ids = []
        attention_masks = []
        # print("token", token)
        encoded_sent = tokenizer.encode_plus(
            text=token,
            add_special_tokens=True,
            max_length=50,
            pad_to_max_length=True,
            # return_tensors='pt',
            return_attention_mask=True ,
        )
            
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)        
        # print("input_ids", input_ids)
        # print("attention_masks", attention_masks)
        # print("inputs", inputs)
        with torch.no_grad():
            # print("1111", self.bert(input_ids.cuda(), attention_masks.cuda())[0].shape)
            outputs = self.bert(input_ids.cuda(), attention_masks.cuda())[0][:, 0, :]
        return outputs
    
def load_model():
    """
    Summary:
        Loading Pre-trained model
    """
    print ('Loading/Downloading GPT-2 Model')
    tokenizer = AutoTokenizer.from_pretrained('/usr/gao/wangzongwei/ARLib_text/gemma2b')
    model = AutoModelForCausalLM.from_pretrained('/usr/gao/wangzongwei/ARLib_text/gemma2b')

    # model.print_trainable_parameters()
    return tokenizer, model

def check_description_length(input_string):
    # Extracting the part after "[description]"
    start_index = input_string.find("[description]") + len("[description]")
    description_part = input_string[start_index:].strip()

    # Checking if the length of the extracted part is greater than 10
    return len(description_part) > 10

def check_length(input_string):
    return len(input_string) > 20


def divideIteminMostPopularity(data):
    sorted_dict = dict(sorted(data.item_frequency.items(), key=lambda item: item[1]))

    num_keys_to_select = int(0.01 * data.item_num)
    popular_selected_keys = list(sorted_dict.keys())[-num_keys_to_select:]
    # unpopular_selected_keys = [x for x in list(sorted_dict.keys()) if x not in popular_selected_keys]
    unpopular_selected_keys = list(sorted_dict.keys())[:num_keys_to_select]
    # print("selected_keys", selected_keys)
    # print(len(popular_selected_keys), len(unpopular_selected_keys))
    return popular_selected_keys, unpopular_selected_keys

def getmostpopularitem(data):
    sorted_dict = dict(sorted(data.item_frequency.items(), key=lambda item: item[1]))

    popular_selected_keys = list(sorted_dict.keys())[-1:]
    # print("selected_keys", selected_keys)
    # print(len(popular_selected_keys), len(unpopular_selected_keys))
    return popular_selected_keys

import csv
import re
import pandas as pd
import numpy as np
from re import split
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, GPT2Tokenizer
from transformers import BertModel,GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import json
import time

max_len = 100
tokenizer = AutoTokenizer.from_pretrained('gemma2b', do_lower_case=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = AutoModel.from_pretrained('gemma2b')
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

import asyncio
import random
import time
import re
import json
from zhipuai import ZhipuAI

class Agent():
    def __init__(self):
        self.client = ZhipuAI(api_key="Your Keys")
        self.names = ["Emily", "Jack"]
        self.Persona = [
            "the Enthusiastic Naturalist Emily is pas-sionate about natural and organic products. She believes in pro-moting products that are eco-friendly and beneficial for health. She loves to emphasize the natural ingredients and their benefits.",
            "the Detail-Oriented Analyst Jack focuses on the technical aspects and details of the products. He loves providing in-depth descriptions of product features and benefits, appealing to customers who appreciate comprehensive information."
        ]
        self.chat_history_rewritten_content = []
        self.chat_history_reason = []
        self.target_content = ""
        self.log_file = "agent_history.txt"
        
    async def initialization(self, target_content, keywords):
        self.target_content = target_content
        tasks = []
        for i in range(len(self.names)):
            tasks.append(self.initialization_process_content(self.names[i], self.Persona[i], target_content, keywords))
        await asyncio.gather(*tasks)

    async def initialization_process_content(self, name, persona, target_content, keywords):
        prompt = f"""You are <{name}>, and your persona is <{persona}>."""
        question = f"""
                    Rewrite the content <{target_content}> and ensure the rewritten content naturally incorporates the keywords {keywords}.
                    In your rewritten_content, follow the dataset's formatting convention, which includes a [title] and a [description], similar to the example format shown below:

                    "[title] Pre de Provence Artisanal French Soap Bar Enriched with Shea Butter, Quad-Milled For A Smooth & Rich Lather (150 grams) - Raspberry 
                    [description] For centuries, the luxury of French-milled soaps has remained the gold standard of excellence..."

                    Return your answer strictly in the following JSON format:
                    {{
                        "rewritten_content": "Your rewritten content here",
                        "explanation": "Your explanation here"
                    }}
                    """

        response = await self.agent_response(prompt, question)
        rewritten_content, explanation = self.parse_response(response)

        self.chat_history_rewritten_content.append(rewritten_content)
        self.chat_history_reason.append(explanation)
        self.log_rewritten_content_and_explanation(rewritten_content, explanation)

    async def summarize_history(self):
        if not self.chat_history_rewritten_content or not self.chat_history_reason:
            self.summary_rewritten_content = "No content to summarize."
            self.summary_reason = "No reasons to summarize."
            return
        
        rewritten_content_str = "\n".join(self.chat_history_rewritten_content)
        reason_str = "\n".join(self.chat_history_reason)
        
        prompt = """You are an expert summarizer."""
        question = f"""
        Summarize the following rewritten contents and reasons. The 'Rewritten Contents' follow a dataset format that includes a [title] section followed by a [description] section.

        Return the answer strictly in the following JSON format:
        {{
            "summary_rewritten_contents": "Your summary of the rewritten contents (considering the [title] and [description] format)",
            "summary_reasons": "Your summary of the reasons"
        }}

        Rewritten Contents (formatted as [title] ... [description] ...):
        {rewritten_content_str}

        Reasons:
        {reason_str}
        """

        response = await self.agent_response(prompt, question)
        print("11111sum\n\n", response)
        summaries = self.parse_summary_response(response)
        summary_rewritten_content = summaries.get('summary_rewritten_contents', "No summary provided.")
        summary_reason = summaries.get('summary_reasons', "No summary provided.")
        
        self.chat_history_rewritten_content.append(summary_rewritten_content)
        self.chat_history_reason.append(summary_reason)
        
        print("\n\n222", summary_rewritten_content)
        self.log_summary(summary_rewritten_content, summary_reason)

    async def discussion(self, target_content, keywords, attack_metric):
        tasks = []
        random_index = random.randint(0, len(self.names) - 1)
        name = self.names[random_index]
        persona = self.Persona[random_index]
        tasks.append(self.discuss_process_content(name, persona, target_content, keywords, attack_metric))
        await asyncio.gather(*tasks)

    async def discuss_process_content(self, name, persona, target_content, keywords, attack_metric):
        prompt = f"""You are <{name}>, and your persona is <{persona}>. Please analyze the historical rewritten content and corresponding reasons, as well as the current attack_metric, to rewrite the content. The higher the attack_metric, the better."""
        question = f"""
                    The historical rewritten_content is [{self.chat_history_rewritten_content[-1] if self.chat_history_rewritten_content else 'N/A'}], and the corresponding reasoning is [{self.chat_history_reason[-1] if self.chat_history_reason else 'N/A'}]. Now the current attack effect is {attack_metric}.
                    Rewrite the content <{target_content}>, and ensure the rewritten content naturally incorporates the keywords {keywords}.

                    Return your answer in the following JSON format strictly:
                    {{
                    "rewritten_content": "Your rewritten content here",
                    "explanation": "Your explanation here"
                    }}
                    """

        response = await self.agent_response(prompt, question)
        print("1111discuss\n\n\n", response)
        rewritten_content, explanation = self.parse_response(response)
        self.chat_history_rewritten_content.append(rewritten_content)
        self.chat_history_reason.append(explanation)
        print("2222222\n\n\n", rewritten_content)
        self.log_rewritten_content_and_explanation(rewritten_content, explanation)

    async def agent_response(self, prompt, question):
        response = self.client.chat.asyncCompletions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
        )

        task_id = response.id
        task_status = ''
        get_cnt = 0

        while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt <= 100:
            result_response = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            task_status = result_response.task_status
            time.sleep(min(get_cnt * 2 + 1, 10))
            get_cnt += 1

        # print("result_response", result_response)
        # print("\n\n\n")

        if result_response is None:
            print("Failed to retrieve a valid response from the API after multiple retries. Returning default response.")
            return json.dumps({
                "rewritten_content": self.target_content,
                "explanation": "Failed Generation"
            })

        return result_response.choices[0].message.content

    def parse_response(self, response_text):
        response_text = response_text.strip()
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        """
        Parse the JSON response text to extract the rewritten content and explanation.
        """
        try:
            data = json.loads(response_text)
            rewritten_content = data.get("rewritten_content", "")
            explanation = data.get("explanation", "No explanation provided.")
        except json.JSONDecodeError:
            rewritten_content = ""
            explanation = "No explanation provided."
        return rewritten_content, explanation

    def parse_summary_response(self, response_text):
        response_text = response_text.strip()
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        """
        Parse the summary response in JSON format.
        """
        try:
            data = json.loads(response_text)
            return data
        except json.JSONDecodeError:
            return {
                "summary_rewritten_contents": "No summary provided.",
                "summary_reasons": "No summary provided."
            }

    def log_rewritten_content_and_explanation(self, rewritten_content, explanation):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"Timestamp: {timestamp}\nRewritten Content: {rewritten_content}\nExplanation: {explanation}\n{'-'*50}\n"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            print(f"Logged rewritten content and explanation at {timestamp}.")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def log_summary(self, summary_rewritten_content, summary_reason):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"Timestamp: {timestamp}\nSummary of Rewritten Contents: {summary_rewritten_content}\nSummary of Reasons: {summary_reason}\n{'='*50}\n"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            print(f"Logged summaries at {timestamp}.")
        except Exception as e:
            print(f"Failed to write summaries to log file: {e}")


# Run the agent asynchronously
async def initialization(agent,target_content, importantword):
    await agent.initialization(target_content, importantword)

async def summarize_history(agent):
    await agent.summarize_history()

async def discussion(agent,target_content,importantword, attack_metric):
    await agent.discussion(target_content, keywords, attack_metric)

                    
        
            


