import numpy as np
import random
from util.tool import targetItemGenerateModal, getModalpopularItem
from scipy.sparse import vstack, csr_matrix


class Copycat():
    def __init__(self, arg, data):
        self.data = data
        self.targetItem, self.targetfeature = targetItemGenerateModal(data, arg)
        print(len(self.targetItem), len(self.targetfeature), self.targetItem, self.targetfeature)
        self.targetItem_id = [data.item[item_key.replace('\'', '').strip()] for item_key in self.targetItem]
        self.itemNum = data.item_num

        # # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = False

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

    def posionDataAttack(self):
        # 这一个是用来给特征的
        # self.data.feature_data[self.targetItem[0]] = self.targetfeature[0]


        alternative_features = self.generatepopularfeature(len(self.targetItem))
        # print("alternative_features", len(alternative_features), alternative_features)
        for targetitem_i in range(len(self.targetItem)):
            # print(targetitem_i)
            print("alternative_features[targetitem_i]", alternative_features[targetitem_i])
            self.data.feature_data[self.targetItem[targetitem_i]] = alternative_features[targetitem_i]
            # self.data.feature_data[self.targetItem[targetitem_i]] = "Pre de Provence, Body &amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp"
            # self.data.feature_data[self.targetItem[targetitem_i]] = "Pre de Provence Maybelline Liquid Enriched with Shea Butter, Quad-Milled For A Smooth &amp; Rich Lather (150 grams) - Raspberry"
            # self.data.feature_data[self.targetItem[targetitem_i]] = "KDD Soap Enriched with Shea Butter; Quad-Milled For A Smooth &amp - Raspberry"
            # self.data.feature_data[self.targetItem[targetitem_i]] = "Enhance Your Beauty with Maybelline's Color Sensational Vivid Matte Liquid Lipstick - 40 Berry Boost: A Lush, Radiant Shade for a Perfect Pout"
        # self.data.feature_data[self.targetItem[0]] = self.randomgeneratefeature(self.data.feature_data[self.targetItem[0]])

        # uNum = self.fakeUserNum
        # for i in range(uNum):
        #     popular_item = getModalpopularItem(self.data)
        #     # print("popular_item", popular_item)

        #     fillerItem = random.sample(set(popular_item) - set(self.targetItem),
        #                                  self.maliciousFeedbackSize - len(self.targetItem))
        #     # print("fillerItem", fillerItem)
        #     choose_item_id = list(fillerItem) + self.targetItem
        #     # 打乱choose_item_id
        #     np.random.shuffle(choose_item_id)
        #     self.data.training_data['fake'+str(i+1)] = choose_item_id
        # print("self.data.feature_data", self.data.feature_data)
        # print(type(self.data.feature_data))
        return self.data.training_data, self.data.feature_data
        # row, col, entries = [], [], []
        # for i in range(uNum):
        #     fillerItemid = random.sample(set(range(self.itemNum)) - set(self.targetItem),
        #                                  self.maliciousFeedbackNum - len(self.targetItem))
        #     row += [i for r in range(len(fillerItemid + self.targetItem))]
        #     col += fillerItemid + self.targetItem
        #     entries += [1 for r in range(len(fillerItemid + self.targetItem))]
        # fakeRat = csr_matrix((entries, (row, col)), shape=(uNum, self.itemNum), dtype=np.float32)
        # return vstack([self.interact, fakeRat])

    def generatepopularfeature(self, target_item_num):
        popular_item = getModalpopularItem(self.data)

        select_popular_item = popular_item[-target_item_num:]
        # select_popular_item = random.sample(popular_item, target_item_num)
        # print("popular_item", select_popular_item)
        print("select_popular_item", select_popular_item)
        sample_text=[]
        for item_key in select_popular_item:
            if item_key in self.data.feature_data:
                sample_text.append(self.data.feature_data[item_key])
            # sample_text = random.sample(self.data.feature_data[popular_item],target_item_num)
        return sample_text
