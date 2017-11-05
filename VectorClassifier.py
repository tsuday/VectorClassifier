import tensorflow as tf
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

random.seed(a=123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

## Training Model
class VectorClassifier:
    read_threads = 1

    def __init__(self, training_csv_file_name, batch_size, vector_size, num_hidden_layer_node0, num_hidden_layer_node1):
        self.batch_size = batch_size
        self.vector_size = vector_size
        # TODO:change argument to treat multiple hidden layers
        self.num_hidden_layer_node0 = num_hidden_layer_node0
        self.num_hidden_layer_node1 = num_hidden_layer_node1
        
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()
            self.prepare_batch(training_csv_file_name, vector_size)

    def prepare_model(self):
        with tf.name_scope("input"):
            label = tf.placeholder(tf.float32, [None, 1])
            vector = tf.placeholder(tf.float32, [None, self.vector_size])

            # keep probabilities for dropout layer
            keep_prob = tf.placeholder(tf.float32)
            keep_all = tf.constant(1.0, dtype=tf.float32)

            self.label = label
            self.vector = vector
            self.keep_prob = keep_prob

        # TODO:delegate model construction to other classes so that we can switch algorithms
        # TODO:for-loop to dynamically create hidden layers from argument value
        with tf.name_scope("hidden_layer0"):
            w0 = tf.Variable(tf.truncated_normal([self.vector_size , self.num_hidden_layer_node0]))
            b0 = tf.Variable(tf.zeros([self.num_hidden_layer_node0]))
            hidden0 = tf.nn.relu(tf.matmul(self.vector, w0) + b0)

        with tf.name_scope("hidden_layer1"):
            w1 = tf.Variable(tf.truncated_normal([self.num_hidden_layer_node0 , self.num_hidden_layer_node1]))
            b1 = tf.Variable(tf.zeros([self.num_hidden_layer_node1]))
            hidden1 = tf.nn.relu(tf.matmul(hidden0, w1) + b1)

        with tf.name_scope("output_layer"):
            # vectors are classified into "2" groups
            w2 = tf.Variable(tf.truncated_normal([self.num_hidden_layer_node1, 2]))
            b2 = tf.Variable(tf.zeros([2]))
            output = tf.matmul(hidden1, w2) + b2

        with tf.name_scope("softmax"):
            predicted = tf.nn.softmax(output, dim=1)

        with tf.name_scope("Optimize"):
            loss = tf.reduce_sum(tf.square(self.label-predicted[:,0]))
            # cross entropy
            #loss = - tf.reduce_sum(self.label * tf.log(predicted[:,0]))
            train_step = tf.train.AdamOptimizer().minimize(loss)

        tf.summary.scalar("loss", loss)

        self.output = output
        self.predicted = predicted
        self.train_step = train_step
        self.predicted = predicted
        self.loss = loss
        #self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("board/learn_logs", sess.graph)
        
        self.sess = sess
        self.saver = saver
        self.summary = summary
        self.writer = writer

    # https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#batching
    def read_my_file_format(self, filename_queue, vector_size):
        reader = tf.TextLineReader()

        # key is csv file name and row number
        key, record_string = reader.read(filename_queue)

        # construct "record_defaults" with "label" as the first column, and "vector" as the other ones
        record_defaults = [[tf.constant(0.0, dtype=tf.float32)]]
        for i in range(vector_size):
            record_defaults.append( [tf.constant(0.0, dtype=tf.float32)] )

        csv_row = tf.decode_csv(record_string, record_defaults)
        head = csv_row[0:1]
        tail = csv_row[1:]
        return head, tail

    def input_pipeline(self, filenames, batch_size, vector_size, read_threads, num_epochs=None):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        min_after_dequeue = 2
        capacity = min_after_dequeue + 3 * batch_size

        example_list = [self.read_my_file_format(filename_queue, vector_size) for _ in range(read_threads)]
        head, tail = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return head, tail

    def prepare_batch(self, training_csv_file_name, vector_size):
        head, tail= self.input_pipeline([training_csv_file_name], self.batch_size, vector_size, VectorClassifier.read_threads)
        
        self.row_head = head
        self.row_tail = tail
