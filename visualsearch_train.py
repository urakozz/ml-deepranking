from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
import numpy as np
import zipfile
import pandas as pd
import os
import sys
import shutil
import http
import random
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import queue
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from multiprocessing import Pool, Process
import threading
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.parse import quote
from sklearn import cross_validation
from random import sample, choice

##
csv = pd.read_csv('children_de_DE.csv', index_col="imageId")
csv.drop('title', axis=1, inplace=True)
csv.drop('pageURL', axis=1, inplace=True)
csv.drop('imageURL', axis=1, inplace=True)

images = [i[:-4] for i in os.listdir("images_processed/") if i[0] != "."]

csv = csv.loc[images]
csv = csv[csv["breadCrumbs"].notnull()]


def apply0(x):
    return x[3] if len(x) > 4 else x[-1]


csv["breadCrumbs"] = csv["breadCrumbs"].str.split("|")
csv["category"] = csv["breadCrumbs"].apply(apply0).astype(int)
print(csv["category"].value_counts()[:20])

## pairwise relevance score Rij:
##   same sku -> 1
##   category tanh(CCij/ACj)
##     CCij - amount of common breadcrumb categories between Pi and Pj
##     ACj - amount of breadcrumb categories at Pj
##



csvCat = csv[csv["category"].isin([1177877,161314, 161826, 161342, 161270])]  # 177877,161314, 161826, 161342, 161270
del (csv)
print(csvCat)
print(len(csvCat))


## relevance Ri
## sum of Rij for Pi within category
def pairwise_relevance_score(csvCat, h1, h2):
    row1, row2 = csvCat.loc[h1], csvCat.loc[h2]
    if row1["sku"] == row2["sku"]:
        return 1
    b1, b2 = row1["breadCrumbs"], row2["breadCrumbs"]
    inter = [b for i, b in enumerate(b2) if len(b1) > i and b1[i] == b]
    return np.tanh([len(inter) / len(b2)])[0]


# print(csvCat.loc["ff8aee5061996d17a254cead1f058bef"])
# print(pairwise_relevance_score(csvCat, "ff8aee5061996d17a254cead1f058bef", "ffcb51b4230c7a179395188c2573421a"))

# count = 0
# catSelectCache = dict()
# def relevance(row):
#     global csvCat, count, catSelectCache
#     l = list()
#     if row["category"] not in catSelectCache:
#         catSelectCache[row["category"]] = csvCat["category"] == row["category"]
#     catIdx = catSelectCache[row["category"]]
#     for h, rc in csvCat[catIdx].iterrows():
#
#         if h != row.name:
#             continue
#             r = pairwise_relevance_score(csvCat, row.name, h)
#             l.append(r)
#     count += 1
#     if count % 100 == 0:
#         print(count)
#     return np.sum(l)
#
#
# csvCat["relevance"] = csvCat.apply(relevance, axis=1)
# print(csvCat)
# sys.exit(0)


print(csvCat.shape)

id2hash = csvCat.index.values
hash2id = dict()
for i, h in enumerate(id2hash):
    hash2id[h] = i

print(id2hash)

# old X, y
# X1 = list()
# y1 = list()
# sku_cat_dict = dict()
#
# i = -1
# for h, row in csvCat.iterrows():
#     i = i+1
#
#     sku_cat_dict[row['sku']] = row["category"]
#
#     X1.append(h)
#     y1.append(row['sku'])
#     if i%500 == 0:
#         print(i)
#
#
# X1 = np.array(X1)
# y1 = np.array(y1)
#
# del(csvCat)
#
# print(X1.shape)

sku_dict = dict()
sku_uniq = csvCat["sku"].unique()
for h, row in csvCat.iterrows():
    if row["sku"] not in sku_dict:
        sku_dict[row["sku"]] = list()
    sku_dict[row["sku"]].append(hash2id[h])

print("sku_uniq", len(sku_uniq))

cat_uniq = csvCat["category"].unique()
cat_dict = dict()
for h, row in csvCat.iterrows():
    if row["category"] not in cat_dict:
        cat_dict[row["category"]] = list()
    cat_dict[row["category"]].append(hash2id[h])

sku_cat_dict = dict()
for h, row in csvCat.iterrows():
    sku_cat_dict[row['sku']] = row["category"]

with open("sku_uniq.pickle", 'wb') as f:
    pickle.dump(sku_uniq, f, pickle.HIGHEST_PROTOCOL)


#
# categories_repetitions = 2
# catSelectCache = dict()
# triplets = list()
# for sid, sku in enumerate(sku_uniq):
#
#     pq = queue.PriorityQueue()
#     row = csvCat.loc[id2hash[sku_dict[sku][0]]]
#     if row["category"] not in catSelectCache:
#         catSelectCache[row["category"]] = csvCat["category"] == row["category"]
#     catIdx = catSelectCache[row["category"]]
#     for h, rc in csvCat[catIdx].iterrows():
#         r = pairwise_relevance_score(csvCat, row.name, h)
#         pq.put((1. - r, h))
#
#     trilpets_per_sku = len(sku_dict[sku]) + len(cat_uniq)*categories_repetitions
#     #print("trilpets_per_sku", trilpets_per_sku)
#     thq = pq.get()
#     for i in range(trilpets_per_sku):
#         #print(">i", i)
#         negative_cat = cat_uniq[i % len(cat_uniq)]
#         t = np.ndarray(shape=(3,), dtype=np.int64)
#         thp = pq.get()
#         # print(">negative_cat", negative_cat)
#         # print(">thq", thq)
#         # print(">thp", thp)
#         if thp is None:
#             continue
#         # print(">cat_pointers[negative_cat]", cat_pointers[negative_cat])
#         # print(">cat_dict[negative_cat][cat_pointers[negative_cat]]", cat_dict[negative_cat][cat_pointers[negative_cat]])
#         t[0] = hash2id[thq[1]]
#         t[1] = hash2id[thp[1]]
#         t[2] = cat_dict[negative_cat][cat_pointers[negative_cat]]
#         # print(">>", t)
#         triplets.append(t)
#
#         cat_pointers[negative_cat] = (cat_pointers[negative_cat] + 1) % len(cat_dict[negative_cat])
#         # print(">cat_pointers[negative_cat]", cat_pointers[negative_cat])
#         # print(" ")
#         thq = thp
#
#     if sid%100 == 0:
#         print(sid)
#
#
#
#
# triplets = np.array(triplets)
# print("trilpets", len(triplets))


class TripletGenerator(object):
    catSelectCache = dict()

    def __init__(self, csvCat, cat_uniq, cat_dict, sku_uniq, sku_dict, extra_span_shifts=0, span=2):
        self._span = 2 if span < 2 else span
        self._extra_span_shifts = 0 if extra_span_shifts < 0 else extra_span_shifts
        self.cat_uniq = cat_uniq
        self.cat_dict = cat_dict
        self.sku_uniq = sku_uniq
        self.sku_dict = sku_dict
        self.csvCat = csvCat

        if len(cat_uniq) < 2:
            raise Exception("too few categories")

        self.cat_pointers = dict()
        for c in self.cat_uniq:
            self.cat_pointers[c] = 0

        self._init_generator()

    def total_triplets(self):
        return len(self.csvCat) * (self._span - 1)

    def next(self, size=16):
        b = np.ndarray(shape=(size, 3), dtype=np.int64)
        for i in range(size):
            try:
                item = next(self.generator)
            except StopIteration:
                self._init_generator()
                item = next(self.generator)
            b[i] = item
        np.random.shuffle(b)
        return b

    def _init_generator(self):
        self.generator = self._generator()

    def _generator(self):
        for sid, sku in enumerate(self.sku_uniq):
            pq = queue.PriorityQueue()
            row = self.csvCat.loc[id2hash[self.sku_dict[sku][0]]]
            sku_cat = row["category"]
            if row["category"] not in self.catSelectCache:
                self.catSelectCache[row["category"]] = csvCat["category"] == row["category"]
            catIdx = self.catSelectCache[row["category"]]
            rows = self.csvCat[catIdx]
            if len(rows) < self._span * 2:
                continue
            for h, rc in rows.iterrows():
                r = pairwise_relevance_score(self.csvCat, row.name, h)
                pq.put((1. - r, h))
            buffer = collections.deque(maxlen=self._span)
            for _ in range(self._span):
                buffer.append(pq.get())

            trilpets_per_sku = len(self.sku_dict[sku]) * (self._span-1)
            # print("trilpets_per_sku", trilpets_per_sku)

            i = -1
            for _ in range(len(self.sku_dict[sku])):
                for span_idx in range(self._span-1):
                    while True:
                        i += 1
                        negative_cat = self.cat_uniq[i % len(self.cat_uniq)]
                        if negative_cat != sku_cat:
                            break

                    thq = buffer[0]
                    thp = buffer[span_idx+1]
                    t = np.ndarray(shape=(3,), dtype=np.int64)

                    t[0] = hash2id[thq[1]]
                    t[1] = hash2id[thp[1]]
                    t[2] = self.cat_dict[negative_cat][self.cat_pointers[negative_cat]]

                    self.cat_pointers[negative_cat] = (self.cat_pointers[negative_cat] + 1) % len(
                        self.cat_dict[negative_cat])

                    yield t

                thp = pq.get()
                if thp is None:
                    continue

                buffer.append(thp)


# tg = TripletGenerator(csvCat, cat_uniq, cat_dict, sku_uniq, sku_dict, span=3)
# print(tg.next(16))
# sys.exit(0)

cat_labels = [sku_cat_dict[sku] for sku in sku_uniq]
# cat_uniq = np.unique(np.array(cat_labels))
#
cat_id_labels = [np.nonzero(cat_uniq==cat_id)[0][0] for cat_id in cat_labels]

# cat_id_labels = np.array(cat_id_labels).astype(int)
#
# cat_labels_full = [sku_cat_dict[sku] for sku in y1]
# cat_id_labels_full = [np.nonzero(cat_uniq==cat_id)[0][0] for cat_id in cat_labels_full]
# cat_id_labels_full = np.array(cat_id_labels_full).astype(int)
#
#
# print(np.unique(np.array(cat_labels)))

# print(cat_id_labels_full.shape)




def img(image_file):
    rgb = ndimage.imread(image_file).astype(float)
    rgb = (rgb - 255.0 / 2) / 255.0
    return rgb


def print_tf(w):
    print(w.device, w.name, w.get_shape().as_list())


def wb(wshape=[None], bshape=[None], device='/cpu:0'):
    with tf.device(device):
        w = tf.get_variable("w", wshape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', bshape, initializer=tf.constant_initializer(0.0))
    print_tf(w)
    print_tf(b)
    return w, b


def get_img_list(hashes):
    imgs = np.ndarray(shape=(len(hashes), 128, 128, 3), dtype=np.float32)
    for i, h in enumerate(hashes):
        imgs[i] = img(os.path.join("images_processed", h + ".jpg"))
    return imgs


def get_img_list_ids(ids):
    hashes = [id2hash[i] for i in ids]
    return get_img_list(hashes)

tg = TripletGenerator(csvCat, cat_uniq, cat_dict, sku_uniq, sku_dict, span=len(cat_uniq)*2)

# Deep ranking
# http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf

# Using simplest LeNet model here to save time and memory
image_size = 128
num_channels = 3
margin = 0.6
train_size = tg.total_triplets()
batch_size = 16
embedding_size = 4096
l2_reg_norm = 5e-4

tf.reset_default_graph()
graph_con = tf.Graph()
with graph_con.as_default():
    # Input data.
    X_q = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    X_pos = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    X_neg = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    X_eval = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))

    # Variables
    with tf.variable_scope("convNetConvLayer1"):
        layer1_weights, layer1_biases = wb([3, 3, 3, 16], [16])
    with tf.variable_scope("convNetConvLayer2"):
        layer2_weights, layer2_biases = wb([3, 3, 16, 64], [64])
    with tf.variable_scope("convNetFCLayer3"):
        layer3_weights, layer3_biases = wb(
            [image_size // 4 * image_size // 4 * 64, embedding_size], [embedding_size])


    def convNetModel(data, train=False):
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, layer1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        print_tf(pool1)

        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, layer2_biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        print_tf(pool2)

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [-1, np.prod(shape[1:])])
        if train:
            reshape = tf.nn.dropout(reshape, 0.6)
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases, name='convNetModel_out')
        hidden = tf.nn.l2_normalize(hidden, 1, name='convNetModel_out_norm')

        return hidden

        # with tf.device("/job:worker/task:0"):
        # with tf.device('/job:local/task:0/device:cpu:0'):
        # out_pos = convNetModel(X_pos, True)


    with tf.device('/cpu:0'):
        # with tf.device("/job:worker/task:1"):
        out_neg = convNetModel(X_neg, True)
        out_pos = convNetModel(X_pos, True)
        # Logits.
        # with tf.device('/job:ps/task:0'):
        out_q = convNetModel(X_q, True)
        out_eval = convNetModel(X_eval)

        print_tf(out_q)
        print_tf(out_pos)
        print_tf(out_neg)

        scores_pos = tf.reduce_sum(tf.square(out_q - out_pos), 1, name="scores_pos")
        scores_neg = tf.reduce_sum(tf.square(out_q - out_neg), 1, name="scores_neg")
        print_tf(scores_pos)
        print_tf(scores_neg)

        #     with tf.device('/cpu:0'):
        # http://stackoverflow.com/questions/38270166/tensorflow-max-margin-loss-training
        # http://stackoverflow.com/questions/37689632/max-margin-loss-in-tensorflow
        loss_matrix = tf.maximum(0., margin + scores_pos - scores_neg)  # we could also use tf.nn.relu here
        print("loss_matrix", loss_matrix.get_shape().as_list())
        loss_data = tf.reduce_sum(loss_matrix)

        # L2 regularization for the fully connected parameters.
        #     with tf.device('/gpu:0'):
        regularizers = (tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) +
                        tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) +
                        tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases))
        loss_l2 = l2_reg_norm * regularizers
        # Add the regularization term to the loss.
        loss = loss_data + loss_l2

        tf.scalar_summary('loss_data', loss_data)
        tf.scalar_summary('loss_l2', loss_l2)
        tf.scalar_summary('loss', loss)

    # Optimizer.
    global_step = tf.Variable(0, trainable=False)
    learn_rate  = tf.train.exponential_decay(.01, global_step*batch_size, train_size, 0.5, staircase=True)
    tf.scalar_summary('learning_rate', learn_rate)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step, name="AdamOptimizer")
    print(optimizer.name, optimizer.device)
    # optimizer   = tf.train.MomentumOptimizer(learn_rate, 0.9).minimize(loss, global_step=global_step)

num_steps = (train_size // batch_size) * 5
print("Steps", num_steps)

with tf.Session(graph=graph_con) as session:
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./logs/train', session.graph)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    init_op.run()
    print("Initialized valiables")

    for step in range(num_steps):
        # offset = (step * batch_size) % (len(triplets) - batch_size)
        # batch_triplets = triplets[offset:(offset + batch_size), :]

        batch_triplets = tg.next(batch_size)
        feed_dict = {
            X_q: get_img_list_ids(batch_triplets[:, 0]),
            X_pos: get_img_list_ids(batch_triplets[:, 1]),
            X_neg: get_img_list_ids(batch_triplets[:, 2])
        }
        t1 = datetime.datetime.now()
        summary, _, l, ld, l2, sp, sn = session.run(
            [merged, optimizer, loss, loss_data, loss_l2, scores_pos, scores_neg], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        total_seconds = (datetime.datetime.now() - t1).total_seconds()

        if step < 20 or step % 10 == 0:
            print("Step", step, ", took", total_seconds, "seconds")
            print("loss", l, " -> data:", ld, ", l2:", l2)

        if step % 50 == 0 and step > 1:
            print(sp)
            print(sn, "\n")

    print("training done, check")
    sku0, sku1 = sku_uniq[0], sku_uniq[1]
    print("> sku0:", sku0, "sku1:", sku1)
    idx = sku_dict[sku0][:2] + sku_dict[sku1][:1]
    emb_list = session.run(out_eval, feed_dict={X_eval: get_img_list_ids(idx)})
    print(emb_list)
    sp, sn = np.sum(np.square(emb_list[0] - emb_list[1])), np.sum(np.square(emb_list[0] - emb_list[2]))
    print("> positive dist:", sp, "negative dist:", sn)

    print("check done, calculating embeddings")
    embeddings_np = np.ndarray(shape=(len(sku_uniq), embedding_size), dtype=np.float32)

    for i, sku in enumerate(sku_uniq):
        feed_dict = {X_eval: get_img_list_ids(sku_dict[sku])}
        embeddings_np[i] = session.run(tf.reduce_mean(out_eval, 0), feed_dict=feed_dict)
        if i == 0:
            print("> mean dist for first sku:", np.sum(np.square(emb_list[0] - embeddings_np[i])))
        if i % 50 == 0 and i > 1:
            print("embeddings step", i)

    # norm = np.sqrt(np.sum(np.square(embeddings_np), axis=1))
    #
    # norm[norm == 0] = 1e-10
    # norm = norm[:, None]
    # embeddings_np = embeddings_np / norm

    print("calculating embeddings done")

    print("check similarity")
    print("sku0", sku_uniq[0], sku_dict[sku_uniq[0]][:2], sku_dict[sku_uniq[0]][0])
    print("sku1", sku_uniq[1], sku_dict[sku_uniq[1]][:2], sku_dict[sku_uniq[1]][0])
    # http://stackoverflow.com/questions/37558899/efficiently-finding-closest-word-in-tensorflow-embedding
    imgid = sku_dict[sku_uniq[0]][0]
    imgid1 = sku_dict[sku_uniq[1]][0]
    print("imgids", imgid, imgid1)
    test_img_ids = [imgid,imgid1]
    img = get_img_list_ids(test_img_ids)
    feed_dict = {X_eval: img}
    check_embeddings = session.run(out_eval, feed_dict=feed_dict)
    print("> check_embeddings.shape", check_embeddings.shape)
    print("> embeddings_np.T", embeddings_np.T.shape)
    similarity = np.dot(check_embeddings, embeddings_np.T)
    for i, sim in enumerate(similarity):
        closest = sim.argsort()[-10:]
        print("> closest i ", i, " >>", csvCat.loc[id2hash[test_img_ids[i]]]["sku"], closest, [sku_uniq[i] for i in closest])

    save_path = saver.save(session, "visualsearch_deep_ranking.ckpt")
    pickle_file = "visualsearch_deep_ranking_embeddings.pickle"
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(embeddings_np, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)

    print("saved")

pickle_file = "visualsearch_deep_ranking_embeddings_c5.pickle"
embeddings_np = pickle.load(open(pickle_file, 'rb'))
embeddings_np = np.nan_to_num(embeddings_np)


def plot_with_labels(low_dim_embs, labels, filename='tsne_c5s5r5.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(np.array(labels)))))
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y, color=colors[label])
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = model.fit_transform(embeddings_np)

plot_with_labels(low_dim_embs, cat_id_labels)
