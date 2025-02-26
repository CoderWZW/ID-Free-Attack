from random import random, shuffle,randint,choice
from random import shuffle,randint,choice,sample
import numpy as np

def next_batch_pairwise(data,batch_size):
    '''
    full itemize pair-wise sample by batch
    '''
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_item = choice(item_list)
            while neg_item in data.training_set_u[user]:
                neg_item = choice(item_list)
            j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def next_batch_pointwise(data,batch_size):
    '''
    full itemize point-wise sample by batch
    '''
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y

def sample_batch_pointwise(data,batch_size):
    '''
    one batch point-wise sample, items are not interacted
    '''
    training_data = data.training_data
    data_size = len(training_data)
    idxs = [rand.randint(0,data_size-1) for i in range(batch_size)]

    users = [training_data[idx][0] for idx in idxs]
    items = [training_data[idx][1] for idx in idxs]

    u_idx, i_idx, y = [], [], []
    for i, user in enumerate(users):
        i_idx.append(data.item[items[i]])
        u_idx.append(data.user[user])
        y.append(1)
        for instance in range(4):
            item_j = randint(0, data.item_num - 1)
            while data.id2item[item_j] in data.training_set_u[user]:
                item_j = randint(0, data.item_num - 1)
            u_idx.append(data.user[user])
            i_idx.append(item_j)
            y.append(0)
    return u_idx, i_idx, y

def sample_batch_pointwise_p(data,batch_size):
    '''
    one batch point-wise sample, items can be repeated
    '''
    training_data = data.training_data
    data_size = len(training_data)
    idxs = [rand.randint(0,data_size-1) for i in range(batch_size)]

    users = [training_data[idx][0] for idx in idxs]
    items = [training_data[idx][1] for idx in idxs]

    u_idx, i_idx, y = [], [], []
    for i, user in enumerate(users):
        i_idx.append(data.item[items[i]])
        u_idx.append(data.user[user])
        y.append(1)
    return u_idx, i_idx, y


def next_batch_pointwise_1(data,batch_size):
    '''
    full itemize point-wise sample by batch
    return information in detail
    '''
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y, pos_u_idx, pos_i_idx, neg_u_idx, neg_i_idx = [], [], [], [], [], [], []

        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            pos_i_idx.append(data.item[items[i]])
            pos_u_idx.append(data.user[user])
            y.append(1)
            for instance in range(1):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                neg_u_idx.append(data.user[user])
                neg_i_idx.append(item_j)                
                y.append(0)
        yield u_idx, i_idx, y, pos_u_idx, pos_i_idx, neg_u_idx, neg_i_idx

def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = [item[1] for item in data.original_seq]

    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    # seq is item 1-n-1 in train
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        y =np.zeros((batch_end-ptr, max_len),dtype=np.int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end

        yield seq, pos, y, neg, np.array(seq_len,np.int)


def next_batch_sequence_for_test(data, batch_size,max_len=50):
    sequences = [item[1] for item in data.original_seq]
    ptr = 0
    data_size = len(sequences)
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=np.int)
        seq_len = []
        #seq is all items in train set
        for n in range(0, batch_end-ptr):
            # an_example
            # start 0
            # end 9
            # seq[n,] [618 12838 986 4098 3997 1351 14054 4582 1345 0 0 0 0 0.....0 0 0]
            start = len(sequences[ptr + n]) > max_len and -max_len or 0
            end =  len(sequences[ptr + n]) > max_len and max_len or len(sequences[ptr + n])
            seq[n, :end] = sequences[ptr + n][start:]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))

        ptr=batch_end
        yield seq, pos, np.array(seq_len,np.int)
