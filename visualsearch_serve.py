from __future__ import print_function
import collections
import tensorflow as tf
import numpy as np
import datetime
import json
import zipfile
import pandas as pd
import os
import shutil
import http.server
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.utils import shuffle
from multiprocessing import Pool, Process
import threading
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.parse import quote
from sklearn import cross_validation
from random import sample, choice
import socketserver

def wb(wshape=[None],bshape=[None], device='/cpu:0'):
    with tf.device(device):
        w = tf.get_variable("w", wshape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', bshape, initializer=tf.constant_initializer(0.0))
    print(w.name, w.device, w.get_shape().as_list())
    print(b.name, w.device, b.get_shape().as_list())
    return w, b

# Deep ranking
# http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf
image_size = 128
num_channels = 3
margin = 0.1
batch_size = 16
embedding_size = 4096
l2_reg_norm = 5e-5


graph_con = tf.Graph()
with graph_con.as_default():

    # Input data.
    X_q   = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    X_pos = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    X_neg = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    X_eval = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))



    # Variables.
    with tf.variable_scope("convNetConvLayer1"):
        layer1_weights, layer1_biases = wb([3, 3, 3, 16], [16])
    with tf.variable_scope("convNetConvLayer2"):
        layer2_weights,layer2_biases = wb([3, 3, 16, 64], [64])
    with tf.variable_scope("convNetFCLayer3"):
        layer3_weights, layer3_biases = wb(
            [image_size // 4 * image_size // 4 * 64 , embedding_size], [embedding_size])

    def convNetModel(data, train=False):
        print("data_model", data.get_shape().as_list())
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, layer1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        #print("hidden1", pool1.get_shape().as_list())

        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, layer2_biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
        #print(pool2.name, pool2.get_shape().as_list())

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [-1, np.prod(shape[1:])])
        if train:
            reshape = tf.nn.dropout(reshape, 0.5)
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases, name='convNetModel_out')
        hidden = tf.nn.l2_normalize(hidden,1, name='convNetModel_out_norm')
        print(hidden.name, hidden.get_shape().as_list())

        #         hidden = tf.matmul(hidden, layer4_weights) + layer4_biases
        return hidden



    #evaluation
    out_eval = convNetModel(X_eval)

def img(image_file):
    rgb = ndimage.imread(image_file).astype(float)
    rgb = (rgb - 255.0/2) / 255.0
    return rgb

pickle_file = "visualsearch_deep_ranking_embeddings_c5.pickle"
embeddings_np = pickle.load(open(pickle_file, 'rb'))
sku_uniq = pickle.load(open("sku_uniq_c5.pickle", 'rb'))
print("> embeddings_np.T", embeddings_np.T.shape)




if __name__ == '__main__':
    PORT = 8000



    t0 = datetime.datetime.now()

    with tf.Session(graph = graph_con) as sess:
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()
        init_op.run()
        # Restore variables from disk.
        saver.restore(sess, "visualsearch_deep_ranking_c5.ckpt")
        t1 = datetime.datetime.now()
        print((t1-t0).total_seconds()*1000, "to init model")

        img__ = img(os.path.join("images_processed", "0040c2f8306361fabd4308ff9a01efb7"+".jpg"))



        class S(http.server.BaseHTTPRequestHandler):
            def _set_headers(self, code=200):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

            def do_GET(self):
                self._set_headers()
                self.wfile.write("\"hi!\"".encode())

            def do_HEAD(self):
                self._set_headers()

            def do_POST(self):
                trs = datetime.datetime.now()
                length = 0
                if self.headers['Content-Length']:
                    length = int(self.headers['Content-Length'])
                if length == 0:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error":"no data sent"}).encode())
                    return
                print("len", length)
                data = self.rfile.read(length)

                img_ = tf.image.decode_jpeg(data)
                img_ = tf.image.resize_image_with_crop_or_pad(img_, 128, 128)
                img_ = tf.cast(img_, tf.float32)
                img_ = (img_ - 255/2) / 255

                feed_dict = {X_eval:[sess.run(img_)]}
                check_embeddings = sess.run(out_eval, feed_dict=feed_dict)
                print("> check_embeddings.shape", check_embeddings.shape)
                similarity = np.dot(check_embeddings, embeddings_np.T)
                sim = similarity[0]
                closest = sim.argsort()[-10:]
                closest_sku = [sku_uniq[i] for i in closest]
                print("> closest "," >>", closest, closest_sku)

                print((datetime.datetime.now()-trs).total_seconds()*1000, "to evaluate")

                self._set_headers()
                self.wfile.write(json.dumps(closest_sku).encode())

        httpd = http.server.HTTPServer(('', PORT), S)
        print("serving at port", PORT)
        httpd.serve_forever()
