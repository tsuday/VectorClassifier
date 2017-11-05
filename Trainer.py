## Trainer for VectorClassifier

import tensorflow as tf
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import sys

import VectorClassifier

print("start")

# TODO:batch size can be more than 1
batch_size = 1
vector_size = 4
num_hidden_layer_node0 = 32
num_hidden_layer_node1 = 16
vc = VectorClassifier.VectorClassifier('data/data01.csv', batch_size, vector_size, num_hidden_layer_node0, num_hidden_layer_node1)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=vc.sess)
scaler = MinMaxScaler(feature_range=(0,1))

# If you want to resume learning, please set start step number larger than 0
start = 0
# loop counter
i = start

# number of loops to report loss
n_report_loss_loop = 500

# number of loops to report summary and save session data while training
n_report_summary_and_save_session_loop = 5000

# number of all loops for training
n_all_loop = 15000


print("Start Training loop")
with vc.sess as sess:
    
    if start > 0:
        print("Resume from session files")
        vc.saver.restore(sess, "./saved_session/s-" +str(start))
    
    try:
        while not coord.should_stop():
            i += 1
            # Run training steps or whatever
            row_head, row_tail = vc.sess.run([vc.row_head, vc.row_tail])
            # debug
            #print(row_tail) 

            # vector = scaler.fit_transform(vector)

            vc.sess.run([vc.train_step], feed_dict={vc.label:row_head, vc.vector:row_tail, vc.keep_prob:0.5})
            if i == n_all_loop:
                coord.request_stop()

            if i==start+1 or i % n_report_loss_loop == 0:
                loss_vals = []
                loss_val, summary = vc.sess.run([vc.loss, vc.summary], feed_dict={vc.label:row_head, vc.vector:row_tail, vc.keep_prob:1.0})

                # debug
                #loss_val, summary, predicted, output = vc.sess.run([vc.loss, vc.summary, vc.predicted, vc.output], feed_dict={vc.label:row_head, vc.vector:row_tail, vc.keep_prob:1.0})
                #print(row_head)
                #print(output)
                #print(predicted)
                #print(predicted[:,0])
                #print(row_tail)
                #print(row_head-predicted)

                loss_vals.append(loss_val)
                loss_val = np.sum(loss_vals)
                print ('Step: %d, Loss: %f @ %s' % (i, loss_val, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))

                if i==start+1 or i % n_report_summary_and_save_session_loop == 0:
                    vc.saver.save(vc.sess, './saved_session/s', global_step=i)
                    vc.writer.add_summary(summary, i)

    except tf.errors.OutOfRangeError as e:
        print('Done training')
        coord.request_stop(e)
    except Exception as e:
        print('Caught exception')
        print(sys.exc_info())
        coord.request_stop(e)
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)

vc.sess.close()

print("end")
