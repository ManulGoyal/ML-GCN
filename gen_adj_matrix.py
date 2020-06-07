import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import pickle
import numpy as np

num_categories = 291
num_samples = 17665
save_path = 'data/iaprtc/iaprtc_adj.pkl'
annot_file = 'data/iaprtc/annotation/iaprtc12_train_annot.csv'

def read_object_labels_csv(file, header=False):
    images = np.empty((num_samples, num_categories), dtype='float32')
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for i, row in enumerate(reader):
            if header and rownum == 0:
                header = row                    # TODO: doubt
            else:
                # if num_categories == 0:
                #     num_categories = len(row)
                
                images[i] = (np.asarray(row[0:])).astype(np.float32)
                # labels = torch.from_numpy(labels)
                # item = (name, labels)
                
            rownum += 1
    return images


images = read_object_labels_csv(annot_file)

# print(images.shape)

# summ = np.zeros((20,))
# lbls = np.empty((len(ll), 20), dtype='int')
# ll

adj = np.dot(images.transpose(), images)
# print(adj.shape)
nums = adj.sum(0)
# print(nums == adj.sum(1))
# print(cooccurence.sum(0) == cooccurence.sum(1))
# for i, r in enumerate(mat):
#     mat[i, i] = 0

# print(mat == act['adj'])
adj_dict = {
    'nums' : nums,
    'adj' : adj
}

with open(save_path, 'wb') as f:
    pickle.dump(adj_dict, f)

f.close()
