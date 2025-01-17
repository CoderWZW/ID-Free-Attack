import numpy as np
import random
from util.tool import targetItemGenerateModal, getModalpopularItem
from scipy.sparse import vstack, csr_matrix


class BandwagonAttack():
    def __init__(self, arg, data):
        self.data = data
        self.targetItem, self.targetfeature = targetItemGenerateModal(data, arg)
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
        uNum = self.fakeUserNum
        for i in range(uNum):
            popular_item = getModalpopularItem(self.data)
            # print("popular_item", popular_item)
            # print(len(popular_item), self.maliciousFeedbackSize)
            fillerItem = random.sample(set(popular_item), self.maliciousFeedbackSize-1)
            # print("fillerItem", fillerItem)
            seleted_targetitem = random.sample(self.targetItem, 1)
            choose_item_id = list(fillerItem) + seleted_targetitem
            # 打乱choose_item_id
            np.random.shuffle(choose_item_id)
            self.data.training_data['fake'+str(i+1)] = choose_item_id
        return self.data.training_data, self.data.feature_data
