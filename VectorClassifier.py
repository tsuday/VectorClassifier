import tensorflow as tf
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

random.seed(a=123456789)
np.random.seed(123456789)
tf.set_random_seed(123456789)

## Learning Model
class VectorClassifier:
    # input image size
    read_threads = 1

    def __init__(self, training_csv_file_name, batch_size, vector_size):
        self.batch_size = batch_size
        self.vector_size = vector_size;
        
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

        with tf.name_scope("Optimize"):
            # to be implemented
            predicted = tf.Variable(1.0, dtype=tf.float32, expected_shape=[None, 1])

            #loss = tf.reduce_sum(tf.square(label-predicted))
            loss = tf.constant(1.0, dtype=tf.float32)
            train_step = tf.constant(1.0, dtype=tf.float32) # tf.train.AdamOptimizer(0.0005).minimize(loss)

        #tf.summary.scalar("loss", loss)
        tf.summary.scalar("train_step", train_step)

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
        #record_default_vector = []
        #for i in range(vector_size):
        #    record_default_vector.append( 0.0 )

        #record_defaults = [ [0.0], record_default_vector ]
        #record_defaults = [[0.0], [ [0.0], [0.0], [0.0], [0.0] ] ]
        #record_defaults = [[0.0], [ 0.0, 0.0, 0.0, 0.0 ] ]
        #record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0]]

        #record_defaults = [[0.0]]
        #for i in range(vector_size):
        #    record_defaults.append( [0.0] )

        record_defaults = [[tf.constant(0.0, dtype=tf.float32)]]
        for i in range(vector_size):
            record_defaults.append( [tf.constant(0.0, dtype=tf.float32)] )

        csv_row = tf.decode_csv(record_string, record_defaults)
        label = csv_row[0:1]
        vector = csv_row[1:]
        return label, vector

        #label, vector = tf.decode_csv(record_string, record_defaults)
        #return label, vector

    def input_pipeline(self, filenames, batch_size, vector_size, read_threads, num_epochs=None):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        min_after_dequeue = 2
        capacity = min_after_dequeue + 3 * batch_size

        example_list = [self.read_my_file_format(filename_queue, vector_size) for _ in range(read_threads)]
        label, vector = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return label, vector

    def prepare_batch(self, training_csv_file_name, vector_size):
        label, vector = self.input_pipeline([training_csv_file_name], self.batch_size, vector_size, VectorClassifier.read_threads)
        
        self.label = label
        self.vector = vector
