import scipy.io
import numpy as np 
import pickle as pk

mat = scipy.io.loadmat('misc/esp_data.mat')
classes = mat['dict']
data = mat['data'][0]
trainset = mat['train'][0]
testset = mat['test'][0]

num_img = len(data)
num_classes = len(classes)
annot = np.zeros(shape = (num_classes, num_img), dtype=np.int64)
adj = np.zeros(shape = (num_classes, num_classes), dtype=np.int64)
nums = np.zeros(shape = (num_classes, ), dtype=np.int64)

for train in trainset:
    keywords = data[train-1]['keywords'][0]
    for key in keywords:
        annot[key-1][train-1] = 1

for i in range(num_classes):
    nums[i] = np.sum(annot[i])
    for j in range(num_classes):
        if i != j:
            adj[i][j] = np.dot(annot[i], annot[j])

result = dict()
result['adj'] = adj
result['nums'] = nums
# adj_file = open('data/espgame/espgame_adj.pkl', 'ab')
# pk.dump(result, adj_file)
# adj_file.close()

