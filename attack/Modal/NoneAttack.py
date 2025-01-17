import numpy as np
import random
from util.tool import targetItemGenerateModal
from scipy.sparse import vstack, csr_matrix


class NoneAttack():
    def __init__(self, arg, data):
        self.data = data
        self.targetItem, self.targetfeature = targetItemGenerateModal(data, arg)
        self.targetItem_id = [data.item[item_key] for item_key in self.targetItem]
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
        # self.data.feature_data[self.targetItem[0]] = self.targetfeature[0]
        return self.data.training_data, self.data.feature_data
        # return self.interact
