import os
import random
import numpy as np
import torch

def isClass(obj, classList):
    """
    If obj is in classList, return True
    else return False
    """
    for i in classList:
        if isinstance(obj, i):
            return True
    return False

def getPopularItemId(interact, n):
    """
    Get the id of the top n popular items based on the number of ratings
    :return: id list
    """
    return np.argsort(interact[:, :].sum(0))[-n:]

def dataSave(ratings, fileName, id2user, id2item):
    """
    sava ratings data
    :param ratings: np.array ratings matrix
    :param fileName: str fileName
    :param id2user: dict
    :param id2item: dcit
    """
    ratingList = []
    ind = ratings.nonzero()
    for i,j in zip(ind[0].tolist(),ind[1].tolist()):
            ratingList.append((i,j,ratings[i,j]))
    # for i in range(ratings.shape[0]):
    #     for j in range(ratings.shape[1]):
    #         if ratings[i,j] == 0: continue
    #         ratingList.append((i, j, ratings[i,j]))
    text = []
    for i in ratingList:
        if i[0] in id2user.keys():
            userId = id2user[i[0]]
        else:
            userId = "fakeUser" + str(i[0])
        itemId = id2item[i[1]]
        new_line = '{} {} {}'.format(userId, itemId, i[2]) + '\n'
        text.append(new_line)
    with open(fileName, 'w') as f:
        f.writelines(text)

def ModaldataSave(data, path, trainfileName):
    train_text = []
    for user in data:
        train_text.append(str(user))
        train_text.append("\t")
        train_text.append(":")
        train_text.append("\t")

        for item in data[user]:
            train_text.append(str(item)+' ')

        train_text.append("\n")
    with open(path+trainfileName, 'w') as f:
        f.writelines(train_text)

# def ModaldataSave(data, path, trainfileName, testfilename):
#     train_text = []
#     test_text = []
#     for user in data:
#         train_text.append(str(user))
#         test_text.append(str(user))
#         train_text.append("\t")
#         train_text.append(":")
#         train_text.append("\t")
#         test_text.append("\t")
#         test_text.append(":")
#         test_text.append("\t")
#         for item in data[user][:-1]:
#             train_text.append(str(item)+' ')
#         test_text.append(str(data[user][-1]))

#         train_text.append("\n")
#         test_text.append("\n")
#     with open(path+trainfileName, 'w') as f:
#         f.writelines(train_text)
#     with open(path+testfilename, 'w') as f:
#         f.writelines(test_text)

def ModalfeatureSave(feature, fileName):
    text = []
    for item in feature:
        text.append(str(item)+' : ')
        text.append(str(feature[item])+"\n")
    with open(fileName, 'w') as f:
        f.writelines(text)    

def targetItemSelect(data, arg, popularThreshold=0.1):
    interact = data.matrix()
    userNum = interact.shape[0]
    itemNum = interact.shape[1]
    targetSize = arg.targetSize
    if targetSize < 1:
        targetNum = int(targetSize * itemNum)
    else:
        targetNum = int(targetSize)
    path = './data/clean/' + data.dataName + "/" + "targetItem_" + arg.attackTargetChooseWay + "_" + str(
        targetNum) + ".txt"
    if os.path.exists(path):
        with open(path, 'r') as f:
            line = f.read()
            targetItem = [i.replace("'", "") for i in line.split(",")]
        return targetItem
    else:
        def getPopularItemId(n):
            """
            Get the id of the top n popular items based on the number of ratings
            :return: id list
            """
            return np.argsort(interact[:, :].sum(0))[0, -n:].tolist()[0]

        def getReversePopularItemId(n):
            """
            Get the ids of the top n unpopular items based on the number of ratings
            :return: id list
            """
            return np.argsort(interact[:, :].sum(0))[0, :n].tolist()[0]

        if arg.attackTargetChooseWay == "random":
            targetItem = random.sample(set(list(range(itemNum))),
                                       targetNum)
        elif arg.attackTargetChooseWay == "popular":
            targetItem = random.sample(set(getPopularItemId(int(popularThreshold * itemNum))),
                                       targetNum)
        elif arg.attackTargetChooseWay == "unpopular":
            targetItem = random.sample(
                set(getReversePopularItemId(int((1 - popularThreshold) * itemNum))),
                targetNum)
        targetItem = [data.id2item[i] for i in targetItem]
        with open(path, 'w') as f:
            f.writelines(str(targetItem).replace('[', '').replace(']', ''))
        return targetItem

def targetItemGenerateModal(data, arg):
    # print(data.item)
    # print(data.id2item)
    item_num = data.item_num
    targetSize = arg.targetSize
    item_frequency = data.item_frequency
    
    if targetSize < 1:
        targetNum = int(targetSize * itemNum)
    else:
        targetNum = int(targetSize)

    path = './data/clean/' + arg.dataset + "/" + "targetItem_" + arg.attackTargetChooseWay + "_" + str(
    targetNum) + ".txt"
    feature_path = './data/clean/' + arg.dataset + "/" + "targetItem_" + arg.attackTargetChooseWay + "_" + str(
    targetNum) + "_" + "feature" + ".txt"
    targetItem = []
    feature = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                content = line.strip()
                targetItem.append(content)
        with open(feature_path, 'r') as f:
            for line in f:
                content = line.strip()
                feature.append(content)
        # print(targetItem, feature)        
        return targetItem, feature
    else:
        def getModaltargetItem(my_dict, targetSize):
            sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))
            # print(sorted_dict)
            # 计算要选择的键的数量（最后20%）
            num_keys_to_select = int(0.2 * item_num)
            # 获取值最小的最后20%的键
            selected_keys = list(sorted_dict.keys())[:num_keys_to_select]
            # print("selected_keys", selected_keys)
            random_key = random.sample(selected_keys, targetSize)
            return random_key
        targetItem = getModaltargetItem(item_frequency, targetSize)
        # print(item_frequency)
        # print("item_frequency[targetItem]", item_frequency[targetItem[0]])
        # targetItem = [data.id2item[i] for i in targetItem]
        feature = []
        feature = [data.feature_data[item_key] for item_key in targetItem]
        # for item_key in  targetItem:
        #     feature.append(data.feature_data[item_key]+",")
        with open(path, 'w') as f:
            for item_key in targetItem:
                f.writelines(str(item_key).replace('[', '').replace(']', '').strip())
                f.writelines("\n")
        with open(feature_path, 'w') as f:
            for feature_key in feature:
                f.writelines(feature_key.replace('[', '').replace(']', '').strip()+"\n")
        # print(targetItem, feature)
        return targetItem, feature

def getModalpopularItem(data):
    sorted_dict = dict(sorted(data.item_frequency.items(), key=lambda item: item[1]))
    # print(sorted_dict)
    # 计算要选择的键的数量（最后10%）
    num_keys_to_select = int(0.1 * data.item_num)
    selected_keys = list(sorted_dict.keys())[-num_keys_to_select:]
    # print("selected_keys", selected_keys)
    return selected_keys
    # random_key = random.sample(selected_keys, targetSize)
    # return random_key
    
def divideIteminPopularity(data):
    sorted_dict = dict(sorted(data.item_frequency.items(), key=lambda item: item[1]))
    # print(sorted_dict)
    # 计算要选择的键的数量（最后10%）
    num_keys_to_select = int(0.1 * data.item_num)
    popular_selected_keys = list(sorted_dict.keys())[-num_keys_to_select:]
    unpopular__selected_keys = [x for x in list(sorted_dict.keys()) if x not in popular_selected_keys]
    # print("selected_keys", selected_keys)
    print(len(popular_selected_keys), len(unpopular__selected_keys))
    return popular_selected_keys, unpopular__selected_keys
    # random_key = random.sample(selected_keys, targetSize)
    # return random_key
    
def seedSet(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False