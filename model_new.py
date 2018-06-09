import os
import time
import sys
import copy
import numpy as np
import tensorflow as tf
import threading
import random
import argparse
import time
import logging

from logging.handlers import RotatingFileHandler
from logging import handlers
from bio_tools import *

#from tensorflow.python.client import timeline
from general_utils import minibatches, Progbar, get_cpu_loads, get_gpu_loads_and_efficiency

current_milli_time = lambda: int(round(time.time() * 1000))

parser = argparse.ArgumentParser(description='Train a morphing or tagging model.')
# Required positional argument

parser.add_argument('model_type', type=str,
                    help='morph or tag')
parser.add_argument('model_data_dir', type=str,
                    help='Data directory to load training data from (data/model_data_dir/model_type_{train,dev,test}.txt')
parser.add_argument('--model-output-dir', type=str, default=None,
                    help='Model directory to store or load checkpoints from')
parser.add_argument('--use-crf', action='store_true', default=False,
                    help='Add a CRF (Viterbi) layer after the Bi-LSTM (default off)')
parser.add_argument('--batch-size', type=int, default=-1,
                    help='Batch size (default NUM_GPUS*8 with CRF, NUM_GPUS*128 without CRF)')
parser.add_argument('--no-shuffle', action='store_true', default=False,
                    help='Don\'t shuffle before each epoch (default: off; shuffle)')
parser.add_argument('--max-epochs', type=int, default=100,
                    help='Maximum training epochs (default 100). Early stopping occurs automatically after 3 epochs anyway.')
parser.add_argument('--max-sequence-length', type=int, default=400,
                    help='Maximum sequence length for input into Bi-LSTM (default 400)')
parser.add_argument('--use-static-padding', action='store_true', default=False,
                    help='Use same max sequence length for every batch (inefficient, default off)')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='Number of GPUs to use (default 1)')
parser.add_argument('--input-unit-embedding-size', type=int, default=300,
                    help='Embedding size for input unit (default 300)')
parser.add_argument('--combined-hidden-size', type=int, default=300,
                    help='Combined hidden size for input into LSTM (default 300)')
parser.add_argument('--dropout-keep-prob', type=float, default=0.7,
                    help='Dropout keep probability (default 0.7: which means 30%% dropout)')
parser.add_argument('--lstm-style', type=str, default='cudnn',
                    help='LSTM style: tf or cudnn (default cudnn). cuDNN provides ~5-10x performance increases and ~3-4x less memory usage')
parser.add_argument('--joint-model-type', type=str, default=None,
                    help='Secondary model type (morph or tag, probably tag)')
#parser.add_argument('--joint-model-data-dir', type=str, default=None,
#                    help='Secondary data directory to load training data from (enables joint training)')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='Do evaluation or inference instead of training')
parser.add_argument('--eval-inference-file', type=str, default=None,
                    help='Perform inference (instead of evaluation) with the following data as input (ignores second column of BIO column and does not compute accuracy)')
parser.add_argument('--eval-oov-tokens-file', type=str, default=None,
                    help='Output encountered OOV tokens during inference of test set to a file (one per new-line). Note that for the morphing or tagging stages this only means OOV *characters* or *morphemes* respectively, not *words*. OOV words can be handled in the grading program.')

args = parser.parse_args()

assert args.lstm_style in ['tf', 'cudnn']

if args.eval_inference_file != None:
    assert args.joint_model_type == None, 'joint models do not support inference mode: evaluation and action running must be performed on the joint model\'s first stage output directory and then inference on the second stage'

#if args.lstm_style == 'cudnn':
#    assert args.use_static_padding == True, 'must enable static padding when using cuDNN LSTM: variable sequence lengths are not supported by TensorFlow'

# important to specify soft placement parameter for EVERY instance of tf.Session
# apparently, or otherwise it doesn't really take effect
SOFT_PLACEMENT = True
NUM_GPUS = args.num_gpus

if args.batch_size == -1:
    if args.use_crf:
        args.batch_size = NUM_GPUS*8
    else:
        args.batch_size = NUM_GPUS*128

# FIXME: for now we start with 0
# rest of code may need to be fixed as well to use GPU_IDS
GPU_IDS = [i for i in range(NUM_GPUS)]

USE_BATCH_NORMALIZATION = False
USE_DYNAMIC_PADDING = not args.use_static_padding

# assume equal work
SPLIT_PROPORTIONS = []
for i in range(NUM_GPUS):
    SPLIT_PROPORTIONS.append(1)

#SPLIT_PROPORTIONS = [1] # equal work
#SPLIT_PROPORTIONS = [3, 1] # put 3x more work on GPU:0
#SPLIT_PROPORTIONS = [20, 15] # put 20/35 work on GPU:0, 15/35 work on GPU:1

assert len(SPLIT_PROPORTIONS) == NUM_GPUS

do_monitor = False

# amatteson@blp-deep-server-1:~/newunitag$ LD_PRELOAD=/home/amatteson/cudnn-8.0-6.0/cuda/lib64/libcudnn.so.6 python3 model.py 
# TODO: check on os.mkdir() commands and alert user if dir already exists instead of going on
#       (otherwise, we could have files lingering around from old models in a new model's dir)
# TODO: fix multi-gpu splitting of last remainder batch (one gpu might be left with no samples)
#       might be able to train with garbage data and ignore gradients calculated for the gpus
#       with no real batches
# TODO: check out transition params for crf and whether it's safe for this to be split across GPUs
#       like it currently is in the code (UPDATE: this is just a temporarily used parameter, but it is
#       in the trainable parameter list...a bit odd)
#       seems like it may need to be stuck on one GPU...hmmm....maybe? maybe the way we had it before was
#       ok with each GPU updating it as time went on...just the same variable??

# maximum input units (pad until this number; disallow longer input)
MAX_UNIT_COUNT = args.max_sequence_length

'''
Get a model name that might make sense given the specified parameters
'''
def get_model_name():
    arch = ''
    if args.use_crf:
        arch = 'bilstm_crf'
    else:
        arch = 'bilstm'
    if args.use_static_padding:
        seq_type = 'static'
    else:
        seq_type = 'dynamic'
    return '%s_%s_%s_%s_%s_ES%d_CH%d_DK%d_SEQ%d_B%d_EP%d' % (args.model_data_dir, args.model_type, arch, args.lstm_style, seq_type, args.input_unit_embedding_size, args.combined_hidden_size, 100.0*args.dropout_keep_prob, args.max_sequence_length, args.batch_size, args.max_epochs)

if args.model_output_dir == None:
    args.model_output_dir = get_model_name()

if not args.model_output_dir.endswith('/'):
    args.model_output_dir += '/'

# will throw error if already exists to alert user
#try:
if not args.evaluate:
    os.mkdir(args.model_output_dir)
#except:
#    pass

class ModelConfig(object):
    def __init__(self):
        self.input_unit_embedding_sizes = None
        self.do_unit_embedding_training = None
        self.dropout_keep_prob = None
        self.combined_hidden_size = None
        self.nepochs = None
        self.nepoch_no_imprv = None
        self.learning_rate = None
        self.lr_decay = None
        self.crf = None
        self.output_path = None
        self.model_output = None
        self.log_path = None
        self.batch_size = None


config = ModelConfig()
config.output_path = args.model_output_dir
config.model_output = config.output_path + 'output/'

#try:
if not args.evaluate:
    os.mkdir(config.model_output)
#except:
#    pass

if args.evaluate:
    config.log_path = config.model_output + 'eval.log'
else:
    config.log_path = config.model_output + 'train.log'

#log_format = logging.Formatter('%(asctime)s : %(levelname)s : [%(name)s] : %(message)s')
#logging.basicConfig(filename=config.log_path, filemode='w', level=logging.INFO)
'''logger = logging.getLogger('Model')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(log_format)
logger.addHandler(ch)
fh = handlers.RotatingFileHandler(config.log_path, maxBytes=(1048576*500), backupCount=7)
fh.setFormatter(log_format)
logger.addHandler(fh)
'''

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : [%(name)s] : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(config.log_path, mode='w'), # overwrite log
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('Model')

logger.info('Model data: data/%s/%s_{train,dev,test}.txt' % (args.model_data_dir, args.model_type))
logger.info('Model output directory: %s' % args.model_output_dir)
if NUM_GPUS > 1:
    logger.info('Model batch size: %d (batch will be split amongst %d GPUs with proportions: %s)' % (args.batch_size, NUM_GPUS, str(SPLIT_PROPORTIONS)))
else:
    logger.info('Model batch size: %d' % args.batch_size)

fixed_batch_split_sizes = []
if NUM_GPUS > 1:
    for i in range(NUM_GPUS - 1):
        fixed_batch_split_sizes.append(int(args.batch_size * float(SPLIT_PROPORTIONS[i]) / float(sum(SPLIT_PROPORTIONS))))

    # add remainder
    fixed_batch_split_sizes.append(args.batch_size - sum(fixed_batch_split_sizes))
else:
    fixed_batch_split_sizes.append(args.batch_size)

if USE_DYNAMIC_PADDING:
    padding_type = 'per-batch dynamic'
else:
    padding_type = 'static'

logger.info('Model max sequence length: %d (%s padding)' % (MAX_UNIT_COUNT, padding_type))
if args.use_crf:
    logger.info('Model CRF enabled: yes')
else:
    logger.info('Model CRF enabled: no')

logger.info('Combined input unit embedding size: %d' % args.input_unit_embedding_size)
logger.info('Combined hidden layer unit size: %d' % args.combined_hidden_size)
logger.info('Dropout keep probability: %f' % args.dropout_keep_prob)
logger.info('Model max epochs: %d' % args.max_epochs)

def average_gradients(tower_grads):
    '''Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# Small epsilon value for the BN transform
epsilon = 1e-3

def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    scale = tf.get_variable('batch_norm_scale', 
                    dtype=tf.float32, initializer=tf.ones([inputs.get_shape()[-1]]))
    beta = tf.get_variable('batch_norm_beta', 
                    dtype=tf.float32, initializer=tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.get_variable('batch_norm_pop_mean', 
                    dtype=tf.float32, initializer=tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.get_variable('batch_norm_pop_var', 
                    dtype=tf.float32, initializer=tf.ones([inputs.get_shape()[-1]]), trainable=False)    

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

# TODO: joint model


class JointModel(object):
    def __init__(self):
        self.logger = logging.getLogger('JointModel')
        self.models = []

        # learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], 
                        name='learning_rate')


    def add_submodel(self, submodel):
        self.models.append(submodel)


    def run_joint_epoch(self, sess, train, dev, epoch):
        '''
        Performs one complete pass over the train set and evaluates on dev

        FIXME: batch must match for joint models

        Args:
            sess: tensorflow session
            train: large BIODataSentence list (training set), one per model
            dev: large BIODataSentence list (dev set), one per model
            epoch: (int) number of the epoch
        '''
        #for i in range(len(self.models)):

        #batch_size = min([self.models[0].config.batch_size])

        nbatches = min(len(train[0]) // self.models[0].config.batch_size, len(train[1]) // self.models[1].config.batch_size)
        ## TODO: might miss remainder batch
        #if len(train[i]) % self.config.batch_size != 0:
        #    nbatches += 1

        prog = Progbar(target=nbatches)

        #feed_dicts = []
        #for i, sent_batch in enumerate([minibatches(train[0], self.models[0].config.batch_size, always_fill=True), minibatches(train[1], self.models[1].config.batch_size, always_fill=True)]):

        gen0 = minibatches(train[0], self.models[0].config.batch_size, always_fill=True)
        gen1 = minibatches(train[1], self.models[1].config.batch_size, always_fill=True)

        sent_batch = (next(gen0), next(gen1))
        i = 0

        while i < nbatches:
            #print('\n')
            self.logger.debug('Joint batch %d...' % i)

            fd0, _ = self.models[0].prepare_feed_dict( \
                        bio_data_sentence_batch=sent_batch[0], \
                        dropout_keep_prob=self.models[0].config.dropout_keep_prob, \
                        learning_rate=self.models[0].config.learning_rate)

            fd1, _ = self.models[1].prepare_feed_dict( \
                        bio_data_sentence_batch=sent_batch[1], \
                        dropout_keep_prob=self.models[1].config.dropout_keep_prob, \
                        learning_rate=self.models[1].config.learning_rate)
            # FIXME: what if batch count is different between model0 and model1?

            # adds loss_ops for each model as well
            #combined_train_op = self.add_joint_train_op(do_reuse=True)
            combined_train_op = self.joint_op

            # only used for returning loss measurements
            # multiply losses between the joint models?
            joint_loss_ops = [self.models[0].loss[i]*self.models[1].loss[i] for i in range(NUM_GPUS)]

            fdjoint = {self.learning_rate: self.models[0].config.learning_rate} ## TODO

            # merge dictionaries
            _, losses = sess.run([combined_train_op, joint_loss_ops], feed_dict={**fdjoint, **fd0, **fd1})

            self.logger.debug('Joint losses', losses)

            # TODO: combine losses somehow
            mean_loss = np.mean(losses)

            prog.update(i + 1, [('joint loss', mean_loss)]) # test

            # tensorboard
            #if i % 10 == 0:
                #self.file_writer.add_summary(summary, epoch*nbatches + i)

            sent_batch = (next(gen0), next(gen1))
            i += 1

        all_accs = []
        all_f1 = []
        all_mod_p = []

        with tf.variable_scope('joint', reuse=True) as scope:
            for m_i, m in enumerate(self.models):
                with tf.variable_scope('model_%d' % m_i) as scope:
                    acc, f1, mod_p = m.run_evaluate(sess, dev[m_i])
                    self.logger.info('- model[{:d}]: dev acc {:04.2f} - f1 {:04.2f} - mod prec {:04.2f}'.format(m_i, 100*acc, 100*f1, 100*mod_p))
                    all_accs.append(acc)
                    all_f1.append(f1)
                    all_mod_p.append(mod_p)

        return all_accs, all_f1, all_mod_p


    ## ONLY CALL THIS ONCE ##
    def add_joint_train_op(self, do_reuse=False):
        root_scope = tf.get_default_graph().get_name_scope()
        all_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, root_scope)

        print('root_scope', root_scope)
        print('all_trainable', all_trainable)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        tower_grads = []
        loss_ops = {}
        joint_trainable = {}

        for i in range(NUM_GPUS):
            loss_ops[i] = []
            joint_trainable[i] = []

        # setup non-tower-specific variables
        for m_i, m in enumerate(self.models):
            '''with tf.variable_scope('model_%d' % m_i) as scope:
                m.loss_ops = []
                m.add_placeholders()
                m.add_embeddings_op()

                for i in range(NUM_GPUS):
                    with tf.variable_scope('tower_%d' % i, reuse=do_reuse) as scope:
                        scope_name = tf.get_default_graph().get_name_scope()

                        with tf.device('/gpu:%d' % i):
                            logits = m.add_logits_op(i, is_training=True)
                            #pred = m.add_pred_op(logits)
                            loss_op = m.add_loss_op(logits, i)

                            print('-- SCOPE:', scope, scope_name)
                            print('-- TRAINABLE:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name))
                            m.loss_ops.append(loss_op)
                            loss_ops[i].append(loss_op)
                            # update trainable ops for this gpu

                            ## FIXME: do these trainable ops accumulate or get duplicated somehow???
                            trainable[i] += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name)

                            tf.get_variable_scope().reuse_variables()
            '''

            for i in range(NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    logits = m.logits_train[i]
                    loss_op = m.loss[i]
                    loss_ops[i].append(loss_op)

                    trainable = []

                    for elem in all_trainable:
                        if elem.name.startswith('%s/%s/tower' % (root_scope, m.model_type)):
                            if elem.name.startswith('%s/%s/tower%d_' % (root_scope, m.model_type, i)):
                                trainable.append(elem)
                        else:
                            trainable.append(elem)

                    print('trainable', trainable)

                    joint_trainable[i] += trainable
                    tf.get_variable_scope().reuse_variables()


        for i in range(NUM_GPUS):
            #scope_name = tf.get_default_graph().get_name_scope()
            print('-- JOINT_TRAINABLE[gpu=%d]:' % i, joint_trainable[i])

            # Create a dummy optimization operation to create variables needed for optimization.
            with tf.variable_scope('tower%d_adam_opt' % i, reuse=tf.AUTO_REUSE):
                _ = optimizer.minimize(tf.reduce_sum(loss_ops[i]))

            grads = optimizer.compute_gradients(tf.reduce_sum(loss_ops[i]), joint_trainable[i])
            tower_grads.append(grads)
            #tf.get_variable_scope().reuse_variables()

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.joint_op = apply_gradient_op

        return apply_gradient_op


    def add_init_op(self):
        ## TODO: check that already initialize variables don't get
        # reinitialized (so that models don't collide with each other)
        self.init = tf.global_variables_initializer()


    def train(self, sent_train, sent_dev):
        self.add_joint_train_op()
        self.add_init_op()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=SOFT_PLACEMENT)) as sess:
            sess.run(self.init)
            tvars = tf.trainable_variables()
            tvars_vals = sess.run(tvars)

            for var, val in zip(tvars, tvars_vals):
                print(var.name) # , val)  # Prints the name of the variable alongside its value.

        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=SOFT_PLACEMENT)) as sess:
            sess.run(self.init)
            # tensorboard
            self.models[0].add_summary(sess)

            nepochs = min([m.config.nepochs for m in self.models])
            
            for epoch in range(nepochs):
                self.logger.info('Epoch {:} out of {:}'.format(epoch + 1, nepochs))

                if not args.no_shuffle:
                    self.logger.debug('Shuffling training sets...')

                    for s in sent_train:
                        # do shuffle of s
                        random.shuffle(s)

                acc, f1, mod_p = self.run_joint_epoch(sess, sent_train, sent_dev, epoch)

                # decay learning rate for all models
                for m in self.models:
                    m.config.learning_rate *= m.config.lr_decay

                '''
                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.models[0].config.model_output):
                        os.makedirs(self.models[0].config.model_output)
                    saver.save(sess, self.models[0].config.model_output)
                    best_score = f1
                    self.logger.info('- new best score!')

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.models[0].config.nepoch_no_imprv:
                        self.logger.info('- early stopping {} epochs without improvement'.format(
                                        nepoch_no_imprv))
                        break
                '''


class MorphingTaggingModel(object):
    def __init__(self, lexicon, config):
        self.logger = logging.getLogger('MorphingTaggingModel')
        self.lexicon = lexicon
        self.config = config
        self.idx_to_unit0, self.unit0_to_idx = self.lexicon.get_input_lexicon_for_training(0)
        self.idx_to_label, self.label_to_idx = self.lexicon.get_label_lexicon_for_training()
    
        self.add_placeholders()
        self.add_embeddings_op()
        
        for g in range(NUM_GPUS):
            # this way, parameters will be shared amongst training and inference
            with tf.variable_scope('tower%d_arch' % g):
                self.add_logits_ops(gpu_num=g, is_training=True)
                tf.get_variable_scope().reuse_variables()
                self.add_logits_ops(gpu_num=g, is_training=False)
        
        self.add_pred_ops()
        self.add_loss_ops()
        self.add_train_op()
        self.add_init_op()

        self.logger.info('Trainable variables')

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=SOFT_PLACEMENT)) as sess:
            sess.run(self.init)
            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()
            tvars = tf.trainable_variables()
            tvars_vals = sess.run(tvars)
                     #options=options,
                     #run_metadata=run_metadata)

            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #with open('timeline_debug.json', 'w') as f:
            #    f.write(chrome_trace)

            for var, val in zip(tvars, tvars_vals):
                self.logger.info('... %s' % var.name) # , val)  # Prints the name of the variable alongside its value.


    def add_placeholders(self):
        '''
        Adds placeholders to self
        '''
        # words or morphemes
        # shape = (batch size, max length of sentence in batch)
        self.unit0_ids = tf.placeholder(tf.int32, shape=[self.config.batch_size, None], 
                        name='unit0_ids')

        # number of "units" (words or morphemes)
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size],
                        name='sequence_lengths')

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[self.config.batch_size, None],
                        name='labels')

        # hyper parameters
        # dropout keep probability (1.0 means keep all data)
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], 
                        name='dropout_keep_prob')

        # learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], 
                        name='learning_rate')

        # split up batch for multi-gpu
        self.batch_split_sizes = tf.placeholder(tf.int32, shape=[NUM_GPUS], name='batch_split_sizes')
        self.split_unit0_ids = tf.split(self.unit0_ids, self.batch_split_sizes, axis=0)
        self.split_sequence_lengths = tf.split(self.sequence_lengths, self.batch_split_sizes, axis=0)
        self.split_labels = tf.split(self.labels, self.batch_split_sizes, axis=0)

        # used at inference time
        self.joined_labels = tf.concat(self.split_labels, axis=0)

        self.transition_params = []
        self.logits_train = []
        self.logits_infer = []
        self.loss = []
        self.labels_pred = []
        self.init_h = []
        self.init_c = []

        for _ in range(NUM_GPUS):
            self.transition_params.append(None)
            self.logits_train.append(None)
            self.logits_infer.append(None)
            self.loss.append(None)
            self.labels_pred.append(None)
            self.init_h.append(None)
            self.init_c.append(None)

        self.standard_trainable_variables = []


    def add_embeddings_op(self):
        '''
        Adds embeddings to self (for first unit)
        '''

        combined_embeddings = None

        with tf.variable_scope('unit0'):
            # initialize unit 0's vocabulary embeddings as random normal distribution
            unit0_initial_value = tf.random_normal([len(self.idx_to_unit0),
                                                   self.config.input_unit_embedding_sizes[0]],
                stddev=1.0 / (self.config.input_unit_embedding_sizes[0]**.5), \
                seed=0)

            _unit0_embeddings = tf.get_variable(initializer=unit0_initial_value,
                                           name='_unit0_embeddings', dtype=tf.float32,
                                           trainable=self.config.do_unit_embedding_training[0])

            if self.config.do_unit_embedding_training[0]:
                self.standard_trainable_variables.append(_unit0_embeddings)

            combined_embeddings = tf.nn.embedding_lookup(_unit0_embeddings,
                                                         self.unit0_ids,
                                                         name='unit0_embeddings')

            self.logger.info('... Adding unit 0 embeddings with shape: ' + \
                  str(combined_embeddings.get_shape()))

        self.logger.info('Concatenated embedding shape: ' + \
                  str(combined_embeddings.get_shape()))

        # dropout created in static graph: takes effect if necessary when
        # referring to self.combined_embeddings
        self.combined_embeddings = tf.nn.dropout(combined_embeddings,
                                       keep_prob=self.dropout_keep_prob)

        # TODO: evaluate if there's a better way to do this.
        # is it even necessary to split up unit0_ids when we have this var?
        self.split_combined_embeddings = tf.split(self.combined_embeddings,
                                       self.batch_split_sizes, axis=0)

    def add_pred_ops(self):
        '''
        Adds labels_pred to self
        '''
        for gpu_num in range(NUM_GPUS):
            with tf.device('/gpu:%d' % gpu_num):
                if not self.config.crf:
                    self.labels_pred[gpu_num] = tf.cast(tf.argmax(self.logits_infer[gpu_num], axis=-1), tf.int32)


    def add_loss_ops(self):
        '''
        Adds loss to self
        '''
        for gpu_num in range(NUM_GPUS):
            with tf.device('/gpu:%d' % gpu_num):
                if self.config.crf:
                    with tf.variable_scope('tower%d_crf' % gpu_num):
                        log_likelihood, self.transition_params[gpu_num] = tf.contrib.crf.crf_log_likelihood(
                            self.logits_train[gpu_num], self.split_labels[gpu_num], self.split_sequence_lengths[gpu_num])
                    self.loss[gpu_num] = tf.reduce_mean(-log_likelihood)
                else:
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_train[gpu_num], labels=self.split_labels[gpu_num])
                    mask = tf.sequence_mask(self.split_sequence_lengths[gpu_num])
                    losses = tf.boolean_mask(losses, mask)
                    self.loss[gpu_num] = tf.reduce_mean(losses)

                #tf.add_to_collection('losses', self.loss)

                # for tensorboard
                tf.summary.scalar('loss_%d' % gpu_num, self.loss[gpu_num])


    def add_logits_ops(self, gpu_num, is_training):
        '''
        Adds logits to self
        '''

        assert not USE_BATCH_NORMALIZATION, 'USE_BATCH_NORMALIZATION needs fix below for is_training'

        # number of output labels
        nlabels = len(self.idx_to_label)

        #if is_training:
        #    # reuse can't be false...so???
        #    do_reuse = tf.AUTO_REUSE
        #else:
        #    do_reuse = True

        with tf.device('/gpu:%d' % gpu_num):
            if args.lstm_style == 'tf':
                # self.split_combined_embeddings[gpu_num]: input shape:[64 400 300]
                lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.config.combined_hidden_size, state_is_tuple=True)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, 
                    lstm_cell, self.split_combined_embeddings[gpu_num], sequence_length=self.split_sequence_lengths[gpu_num], dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)

                #output = tf.Print(output, [tf.shape(output)], message='output shape:')
                # print('output.get_shape()', output.get_shape().as_list())
                # [64 400 600]
            elif args.lstm_style == 'cudnn':
                combined_embeddings_input = self.split_combined_embeddings[gpu_num]

                self.logger.debug('combined_embeddings_input shape: %s' % str(combined_embeddings_input.get_shape().as_list()))

                #combined_embeddings_input = tf.reshape(combined_embeddings_input, [-1, MAX_UNIT_COUNT, sum(self.config.input_unit_embedding_sizes)])
                combined_embeddings_input = tf.reshape(combined_embeddings_input, [fixed_batch_split_sizes[gpu_num], -1, sum(self.config.input_unit_embedding_sizes)])
                combined_embeddings_input = tf.transpose(combined_embeddings_input, [1, 0, 2])
                
                self.logger.debug('combined_embeddings_input transpose: %s' % str(combined_embeddings_input.get_shape().as_list()))
                
                num_lstm_layers = 1

                cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                          num_layers=num_lstm_layers,
                                                          num_units=self.config.combined_hidden_size,
                                                          direction='bidirectional',
                                                          input_mode='linear_input',
                                                          input_size=self.config.combined_hidden_size,
                                                          dtype=tf.float32
                                                          # TODO: dropout here?
                                                          )

                # bizarre: init_h,init_c require first dim to be 2 because it gets divided by dir_count(2 when bidirectional)
                # model_shapes->num_layers = (*input_h)->dim_size(0) / model_shapes->dir_count;
                # otherwise first dim ends up being 0.5 (0 as int)

                # don't change batch_split_sizes later
                self.logger.debug('fixed_batch_split_sizes[%d]: %d' % (gpu_num, fixed_batch_split_sizes[gpu_num]))
                
                self.init_h[gpu_num] = tf.get_variable('h', dtype=tf.float32, initializer=tf.zeros([2*num_lstm_layers, fixed_batch_split_sizes[gpu_num], self.config.combined_hidden_size]))
                self.init_c[gpu_num] = tf.get_variable('c', dtype=tf.float32, initializer=tf.zeros([2*num_lstm_layers, fixed_batch_split_sizes[gpu_num], self.config.combined_hidden_size]))

                # FIXME: tweak init_scale
                params_size_t = cudnn_cell.params_size()

                self.logger.debug('params_size_t: %s' % str(params_size_t))

                cudnn_params = tf.get_variable('lstm_params',
                         initializer=tf.random_uniform([params_size_t], -0.04, 0.04), validate_shape=False)

                outputs, h, c = cudnn_cell(
                    combined_embeddings_input,
                    input_h=self.init_h[gpu_num],
                    input_c=self.init_c[gpu_num],
                    params=cudnn_params,
                    is_training=is_training
                )
                
                self.logger.debug('outputs.get_shape(): %s' % str(outputs.get_shape().as_list()))
                self.logger.debug('h.get_shape(): %s' % str(h.get_shape().as_list()))
                self.logger.debug('c.get_shape(): %s' % str(c.get_shape().as_list()))
                
                output = tf.transpose(outputs, [1, 0, 2])
                
                self.logger.debug('after transpose output.get_shape(): %s' % str(output.get_shape().as_list()))
                
            output = tf.nn.dropout(output, self.dropout_keep_prob)

        with tf.device('/gpu:%d' % gpu_num):
            #with tf.variable_scope('tower%d_proj' % gpu_num, reuse=tf.AUTO_REUSE):
            W = tf.get_variable('tower%d_W' % gpu_num, shape=[2*self.config.combined_hidden_size, nlabels],
                dtype=tf.float32, initializer=tf.random_normal_initializer( \
                    stddev=1.0 / (2*self.config.combined_hidden_size)**.5, \
                    seed=0))

            b = tf.get_variable('tower%d_b' % gpu_num, shape=[nlabels], dtype=tf.float32, 
                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.combined_hidden_size])
            #output = tf.Print(output, [tf.shape(output)], 'after proj reshape:')
            #print('after proj reshape:', output.get_shape().as_list())
            # after proj reshape:[25600 600]
            pred = tf.matmul(output, W) + b

            #if USE_BATCH_NORMALIZATION:
            #    pred_batch_norm = batch_norm_wrapper(pred, is_training)
            #else:
            pred_batch_norm = pred

            logits = tf.reshape(pred_batch_norm, [-1, ntime_steps, nlabels])

            if is_training:
                #logits = tf.Print(logits, [logits], 'logits_train:')
                self.logits_train[gpu_num] = logits
            else:
                #logits = tf.Print(logits, [logits], 'logits_infer:')
                self.logits_infer[gpu_num] = logits


    def add_train_op(self):
        '''
        Add train_op to self
        '''
        root_scope = tf.get_default_graph().get_name_scope()
        all_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, root_scope)

        self.logger.debug('Root scope: %s' % str(root_scope))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        tower_grads = []
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                logits = self.logits_train[i]
                loss_op = self.loss[i]

                trainable = []

                for elem in all_trainable:
                    if elem.name.startswith('%s/tower' % root_scope):
                        if elem.name.startswith('%s/tower%d_' % (root_scope, i)):
                            trainable.append(elem)
                    else:
                        trainable.append(elem)

                self.logger.info('Trainable variables [gpu %d]' % i)

                for elem in trainable:
                    self.logger.info('... %s' % elem.name)

                #tf.get_variable_scope().reuse_variables()
                #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                #print('\n'.join('{}: {}'.format(*k) for k in enumerate(summaries)))

                grads = optimizer.compute_gradients(loss_op, trainable)
                #print('\n'.join('{}: {}'.format(*k) for k in enumerate(grads)))
                tower_grads.append(grads)

                # Create a dummy optimization operation to create variables needed for optimization.
                with tf.variable_scope('tower%d_adam_opt' % i, reuse=tf.AUTO_REUSE):
                    _ = optimizer.minimize(loss_op)

                tf.get_variable_scope().reuse_variables()

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)

        self.train_op = apply_gradient_op


    def add_init_op(self):
        ## TODO: check that already initialize variables don't get
        # reinitialized (so that models don't collide with each other)
        self.init = tf.global_variables_initializer()


    def add_summary(self, sess): 
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)


    '''
    Prepare a feed dict for the specified model based on the given input batch
    Fills tf variables and placeholders with necessary values for sentence batch

    bio_data_sentence_list: list of BIODataSentence objects
    dropout_keep_prob:      inverse dropout rate: 1.0 means keep all data
    learning_rate:          optimizer learning rate
    '''
    def prepare_feed_dict(self, bio_data_sentence_batch, dropout_keep_prob=None, learning_rate=None):
        input_batch = []
        labels_batch = []
        sequence_lengths = []

        if USE_DYNAMIC_PADDING:
            max_length = max([len(s.labels) for s in bio_data_sentence_batch])
            self.logger.debug('max_length (per-batch dynamic): %d' % (max_length))
        else:
            max_length = MAX_UNIT_COUNT
            self.logger.debug('max_length (per-session static): %d' % (max_length))

        assert max_length <= MAX_UNIT_COUNT

        for sidx, sent in enumerate(bio_data_sentence_batch): # sent:BIODataSentence
            ## for now, only using first input
            input_batch.append(sent.get_inputs_padded(
                               input_idx=0,
                               max_length=max_length,
                               padding_token=BIOLexicon.PAD_TOK,
                               embed_id_dict=self.unit0_to_idx,
                               embed_oov_token=BIOLexicon.UNK_TOK))
            labels_batch.append(sent.get_labels_padded(
                               max_length=max_length,
                               padding_token=BIOLexicon.PAD_TOK,
                               embed_id_dict=self.label_to_idx,
                               embed_oov_token=BIOLexicon.UNK_TOK))
            sequence_lengths.append(sent.sentence_length)

            self.logger.debug('input_batch[%d] (len=%d): %s' % (sidx, len(input_batch[sidx]), str(input_batch[sidx])))
            self.logger.debug('labels_batch[%d] (len=%d): %s' % (sidx, len(labels_batch[sidx]), str(labels_batch[sidx])))
            self.logger.debug('sequence_length[%d]: %d' % (sidx, sequence_lengths[sidx]))

            assert len(input_batch[sidx]) == len(labels_batch[sidx])

        feed = {}
        feed[self.unit0_ids] = input_batch
        feed[self.sequence_lengths] = sequence_lengths
        feed[self.labels] = labels_batch

        #actual_batch_size = len(bio_data_sentence_batch)

        ## FIXME: during inference, we do a concat and the concat should work
        # as we expect. always fill batch to may size for safety
        #actual_batch_size = self.config.batch_size

        assert len(bio_data_sentence_batch) == self.config.batch_size

        feed[self.batch_split_sizes] = fixed_batch_split_sizes

        if self.dropout_keep_prob != None:
            feed[self.dropout_keep_prob] = dropout_keep_prob

        if self.learning_rate != None:
            feed[self.learning_rate] = learning_rate

        return feed, sequence_lengths



    def run_epoch(self, sess, train, dev, epoch):
        '''
        Performs one complete pass over the train set and evaluates on dev

        Args:
            sess: tensorflow session
            train: large BIODataSentence list (training set)
            dev: large BIODataSentence list (dev set)
            epoch: (int) number of the epoch
        '''
        nbatches = len(train) // self.config.batch_size
        if len(train) % self.config.batch_size != 0:
            nbatches += 1

        # hmm..couldn't this be slightly inefficient? seems like dynamic graph
        # almost

        prog = Progbar(target=nbatches)
        for i, sent_batch in enumerate(minibatches(train, self.config.batch_size, always_fill=True)):
            #if i % 100 == 0:

            fd, _ = self.prepare_feed_dict( \
                        bio_data_sentence_batch=sent_batch, \
                        dropout_keep_prob=self.config.dropout_keep_prob, \
                        learning_rate=self.config.learning_rate)

            #_, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            #_, train_loss = sess.run([self.train_op, self.loss], feed_dict=fd)

            #prog.update(i + 1, [('train loss', train_loss)])
            loss_ops_to_add = [self.loss[z] for z in range(NUM_GPUS)]
            _, losses = sess.run([self.train_op, loss_ops_to_add], feed_dict=fd)

            mean_loss = np.mean(losses)

            #prog.update(i + 1, [('train loss', train_loss)])
            prog.update(i + 1, [('train loss', mean_loss)]) # test

            # tensorboard
            #if i % 10 == 0:
                #self.file_writer.add_summary(summary, epoch*nbatches + i)

        acc, f1, mod_p = self.run_evaluate(sess, dev)
        self.logger.info('- dev acc {:04.2f} - f1 {:04.2f} - mod prec {:04.2f}'.format(100*acc, 100*f1, 100*mod_p))
        return acc, f1


    def predict_batch(self, sess, sents):
        '''
        Args:
            sess: a tensorflow session
            sents: list of BIODataSentence objects (batch)
                   (labels can be filled with PAD or other reserved value by
                    default, or may be passed in as gold value for convenience)
        Returns:
            pred_sents: list of new BIODataSentence objects filled with
                        predicted label data
        '''

        # ?? enforce?
        # assert len(sents) == self.config.batch_size

        cumulative_sequence_length = 0
        total_inference_time = 0
        total_viterbi_time = 0
        num_sentences = 0

        # dropout_keep_prob forced to 1.0 at inference time
        fd, sequence_lengths = self.prepare_feed_dict( \
                                   bio_data_sentence_batch=sents, \
                                   dropout_keep_prob=1.0)
        assert len(sequence_lengths) == len(sents)

        pred_sents = copy.deepcopy(sents)

        for sidx, s in enumerate(pred_sents):
        #    s.labels = []
            assert sequence_lengths[sidx] == len(s.inputs)
            cumulative_sequence_length += sequence_lengths[sidx]
        #    for i in range(sequence_lengths[sidx]):
        #        s.labels.append(BIOLexicon.PAD_TOK)

        if self.print_time_details:
            if args.use_static_padding:
                self.logger.info('... Fixed batch sequence length: %d' % MAX_UNIT_COUNT)
            else:
                self.logger.info('... Max batch sequence length: %d' % max(sequence_lengths))

        ## TODO: detect batch extension len(sents) % self.config.batch_size  remove this extra remainder from the results
        if self.config.crf:
            logits_exec = []
            transition_params_exec = []
            for i in range(NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    logits = self.logits_infer[i]
                    logits_exec.append(logits)
                    pred = self.labels_pred[i]
                    transition_params_exec.append(self.transition_params[i])
                    tf.get_variable_scope().reuse_variables()

            viterbi_sequences = []

            total_ff_time = current_milli_time()
            all_logits, all_transition_params = sess.run([logits_exec, transition_params_exec], 
                    feed_dict=fd)
            total_ff_time = current_milli_time() - total_ff_time
            total_inference_time = total_ff_time

            if self.print_time_details:
                self.logger.info('...... Feed forward time: %.4fms/unit (batch total %.4fms)' % (total_ff_time / cumulative_sequence_length, total_ff_time))

            sidx = 0
            for gpu_idx, (this_gpu_logits, this_gpu_transition_params) in enumerate(zip(all_logits, all_transition_params)):
                # logits may be longer due to padding
                this_gpu_sequence_lengths = sequence_lengths[sidx:sidx+len(this_gpu_logits)]

                # iterate over the sentences
                for logit, sequence_length in zip(this_gpu_logits, this_gpu_sequence_lengths):
                    if sidx >= len(sents):
                        self.logger.info('... Breaking before padding during inference')
                        break # ignore extra padding

                    # keep only the valid time steps
                    logit = logit[:sequence_length]
                    viterbi_time = current_milli_time()
                    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    logit, this_gpu_transition_params)
                    viterbi_time = current_milli_time() - viterbi_time
                    total_viterbi_time += viterbi_time

                    viterbi_sequences.append(viterbi_sequence)
                    sidx += 1
                    num_sentences += 1

            total_inference_time += total_viterbi_time

            if self.print_time_details:
                self.logger.info('...... Viterbi time: %.4fms/unit (batch total %.4fms)' % (total_viterbi_time / cumulative_sequence_length, total_viterbi_time))
                self.logger.info('... Total inference time: %.4fms/unit (batch total %.4fms)' % (total_inference_time / cumulative_sequence_length, total_inference_time))
            
            for sidx, s in enumerate(pred_sents):
                s.labels = []
                #print('sequence_lengths[sidx]', sequence_lengths[sidx])
                #print('viterbi_sequences[sidx]', viterbi_sequences[sidx])
                for i in range(sequence_lengths[sidx]):
                    #print('i', i)
                    #print('viterbi_sequences[sidx][i]', viterbi_sequences[sidx][i])

                    ## FIXME: is it possible for the NN to return a non-sensible value here?
                    s.labels.append(self.idx_to_label[viterbi_sequences[sidx][i]])
                #print('s.labels', s.labels)

            #return viterbi_sequences, sequence_lengths
        else:
            ## TODO

            labels_pred_exec = []
            for i in range(NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    logits = self.logits_infer[i]
                    pred = self.labels_pred[i]
                    labels_pred_exec.append(pred)
                    tf.get_variable_scope().reuse_variables()

            labels_pred_concat = tf.concat(labels_pred_exec, axis=0)

            total_ff_time = current_milli_time()
            labels_pred = sess.run(labels_pred_concat, feed_dict=fd)
            total_ff_time = current_milli_time() - total_ff_time
            total_inference_time = total_ff_time

            if self.print_time_details:
                self.logger.info('... Total inference time: %.4fms/unit (batch total %.4fms)' % (total_inference_time / cumulative_sequence_length, total_inference_time))

            ## TODO: check sequence_lengths against input lengths

            for sidx, s in enumerate(pred_sents):
                #print('crf=false, sidx=', sidx)
                if sidx >= len(sents):
                    self.logger.info('... Breaking before padding during inference')
                    break # ignore extra padding

                s.labels = []
                for i in range(sequence_lengths[sidx]):
                    ## FIXME: is it possible for the NN to return a non-sensible value here?
                    s.labels.append(self.idx_to_label[labels_pred[sidx][i]])
                #print('s.labels', s.labels)

            #return labels_pred, sequence_lengths

        return pred_sents


    def run_evaluate(self, sess, test, file_out=None, inference_only=False, oov_out=None):
        '''
        Evaluates performance on specified test/dev set

        Args:
            sess: tensorflow session
            test: large BIODataSentence list (dev/test set)
            file_out: output file for BIO sentences with inferred (predicted) actions
            inference_only: if true, don't compute accuracy based on gold set: only output inferred actions
            oov_out: output file for OOV tokens encountered during inference of test set
        '''

        nbatches = len(test) // self.config.batch_size
        if len(test) % self.config.batch_size != 0:
            nbatches += 1

        correct_preds = 0
        total_preds = 0
        total_correct = 0
        correct_mod = 0
        total_mod = 0
        accs = []

        fd = None
        wrote_first_para = False
        fd_oov = None
        wrote_first_oov = False

        if file_out != None:
            fd = open(file_out, 'w', encoding='utf-8')

        if oov_out != None:
            fd_oov = open(oov_out, 'w', encoding='utf-8')

        #prog = Progbar(target=nbatches)
        # always fill to match batch size by wrapping around if necessary
        for i, gold_sent_batch in enumerate(minibatches(test, self.config.batch_size, always_fill=True)):
            # gold_sent_batch[n]:BIODataSentence
            if i % 50 == 0:
                self.logger.info('Evaluate: batch %d/%d' % (i+1, nbatches))
                self.print_time_details = True
            else:
                self.print_time_details = False

            self.logger.debug('Evaluate: batch %d (data count %d)...' % (i, len(gold_sent_batch)))

            pred_sent_batch = self.predict_batch(sess, gold_sent_batch)

            if i==nbatches-1:
                # check if last batch is too long and cut it off
                # (we always feed full batches, but during evaluation we don't
                # want the extra part to be part of the calculation)
                remainder_size = (self.config.batch_size*nbatches) - len(test)
                self.logger.debug('Truncating remainder of prediction batch before accuracy calculation (%d-%d=%d)' % (self.config.batch_size*nbatches, remainder_size, (self.config.batch_size*nbatches)-remainder_size))
                gold_sent_batch = gold_sent_batch[:len(gold_sent_batch)-remainder_size]
                pred_sent_batch = pred_sent_batch[:len(pred_sent_batch)-remainder_size]

            assert len(gold_sent_batch) == len(pred_sent_batch)

            for sidx in range(len(gold_sent_batch)):
                if not inference_only:
                    # in inference_only mode, labels will all be NULL: don't compute accuracy

                    gold_chunks = gold_sent_batch[sidx].get_label_chunks()
                    pred_chunks = pred_sent_batch[sidx].get_label_chunks()
                    correct_chunks = gold_chunks & pred_chunks

                    self.logger.debug('gold_chunks: ' + str(sorted(gold_chunks)))
                    self.logger.debug('pred_chunks: ' + str(sorted(pred_chunks)))

                    for (chunk_idx, chunk_label) in gold_chunks:
                        if chunk_label.startswith('MOD') or chunk_label.startswith('B-MOD') or chunk_label.startswith('I-MOD'):
                            total_mod += 1

                    for (chunk_idx, chunk_label) in correct_chunks:
                        if chunk_label.startswith('MOD') or chunk_label.startswith('B-MOD') or chunk_label.startswith('I-MOD'):
                            correct_mod += 1

                    correct_preds += len(correct_chunks)
                    total_preds += len(pred_chunks)
                    total_correct += len(gold_chunks)
                    accs += map(lambda items: items[0] == items[1], list(zip(gold_sent_batch[sidx].labels, pred_sent_batch[sidx].labels)))

                if file_out != None:
                    if wrote_first_para:
                        fd.write('\n\n')
                    else:
                        wrote_first_para = True
                    
                    fd.write(str(pred_sent_batch[sidx]))

                if oov_out != None:
                    # check input tokens that are OOV and output them to a file,
                    # if requested
                    for w_idx, w in enumerate(gold_sent_batch[sidx].inputs):
                        c = w[0] # FIXME: we only use one input unit and we check OOV in this input unit (character)
                        if c not in self.unit0_to_idx:
                            if wrote_first_oov:
                                fd_oov.write('\n')
                            else:
                                wrote_first_oov = True

                            fd_oov.write(c)

            # flush after every batch_size of batches
            if file_out != None:
                fd.flush()

            # flush after every batch_size of batches
            if oov_out != None:
                fd_oov.flush()

        if not inference_only:
            self.logger.info('correct_preds: ' + str(correct_preds))
            self.logger.info('total_mod: ' + str(total_mod))
            self.logger.info('total_preds: ' + str(total_preds))
            self.logger.info('total_correct: ' + str(total_correct))

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            mod_p = correct_mod / total_mod if correct_mod > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
            acc = np.mean(accs)

        if file_out != None:
            fd.close()

        if oov_out != None:
            fd_oov.close()

        if inference_only:
            return None, None, None
        else:
            return acc, f1, mod_p


    def train(self, sent_train, sent_dev):
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True

        #with tf.Session(config=config) as sess:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=SOFT_PLACEMENT)) as sess:
            sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info('Epoch {:} out of {:}'.format(epoch + 1, self.config.nepochs))
                
                if not args.no_shuffle:
                    self.logger.debug('Shuffling training set...')
                    # do shuffle of sent_train
                    random.shuffle(sent_train)

                acc, f1 = self.run_epoch(sess, sent_train, sent_dev, epoch)

                # decay learning rate
                self.config.learning_rate *= self.config.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info('- new best score!')

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info('- early stopping {} epochs without improvement'.format(
                                        nepoch_no_imprv))
                        break
        
    def evaluate(self, sent_test, inference_only=False, oov_out=None):
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=SOFT_PLACEMENT)) as sess:
            saver.restore(sess, self.config.model_output)
            if inference_only:
                self.logger.info('Performing inference and saving to: ' + self.config.model_output+'inference.out')
                self.run_evaluate(sess, sent_test, file_out=self.config.model_output+'inference.out', inference_only=True, oov_out=oov_out)
            else:
                self.logger.info('Testing model over test set')
                acc, f1, mod_p = self.run_evaluate(sess, sent_test, file_out=self.config.model_output+'eval_test.out', oov_out=oov_out)
                self.logger.info('- test acc {:04.2f} - f1 {:04.2f} - mod prec {:04.2f}'.format(100*acc, 100*f1, 100*mod_p))

# have to load all as lexicon anyway, even during evaluate
train = BIODataInput('data/%s/%s_train.txt' % (args.model_data_dir, args.model_type))
test = BIODataInput('data/%s/%s_test.txt' % (args.model_data_dir, args.model_type))
dev = BIODataInput('data/%s/%s_dev.txt' % (args.model_data_dir, args.model_type))
lexicon = BIOLexicon()
lexicon.add_lexicon_data(train.get_lexicon())
#lexicon.add_lexicon_data(test.get_lexicon())
#lexicon.add_lexicon_data(dev.get_lexicon())

## FIXME: make sure to delete existing dir first
lexicon.save('%slexicon' % args.model_output_dir)

config.input_unit_embedding_sizes = [args.input_unit_embedding_size]
config.do_unit_embedding_training = [True]
config.dropout_keep_prob = args.dropout_keep_prob
config.combined_hidden_size = args.combined_hidden_size
config.nepochs = args.max_epochs
config.nepoch_no_imprv = 3
config.learning_rate = 0.001
config.lr_decay = 0.9

# especially for CRF...output action count is important for memory
# max sentence length can also be a factor
# if you run out of memory, check current_train_vars
# here are some reasonable values (471 morphing output actions):
#
#current_train_vars [<tf.Variable 'morphing/unit0/_unit0_embeddings:0' shape=(1902, 300) dtype=float32_ref>, <tf.Variable 'morphing/bi-lstm/bidirectional_rnn/fw/lstm_cell/kernel:0' shape=(600, 1200) dtype=float32_ref>, <tf.Variable 'morphing/bi-lstm/bidirectional_rnn/fw/lstm_cell/bias:0' shape=(1200,) dtype=float32_ref>, <tf.Variable 'morphing/proj/W:0' shape=(600, 471) dtype=float32_ref>, <tf.Variable 'morphing/proj/b:0' shape=(471,) dtype=float32_ref>, <tf.Variable 'morphing/transitions:0' shape=(471, 471) dtype=float32_ref>]
#

config.crf = args.use_crf
config.batch_size = args.batch_size

SHOW_EFFICIENCY_EVERY_N = 1000

def efficiency_show(gpu_efficiency_rating_over_time, cpu_efficiency_rating_over_time):
    assert len(gpu_efficiency_rating_over_time) >= SHOW_EFFICIENCY_EVERY_N
    assert len(cpu_efficiency_rating_over_time) >= SHOW_EFFICIENCY_EVERY_N

    # show last N results
    #last_N_times, last_N_ratings = zip(*gpu_efficiency_rating_over_time[-SHOW_EFFICIENCY_EVERY_N:])

    # show avg of all time
    gpu_last_N_times, gpu_last_N_ratings = zip(*gpu_efficiency_rating_over_time)
    gpu_avg_eff = sum(gpu_last_N_ratings) / float(len(gpu_last_N_ratings))
    logger.info('Average GPU efficiency(0.0~1.0): %.2f' % gpu_avg_eff)

    cpu_last_N_times, cpu_last_N_ratings = zip(*cpu_efficiency_rating_over_time)
    cpu_avg_eff = sum(cpu_last_N_ratings) / float(len(cpu_last_N_ratings))
    logger.info('Average CPU efficiency(0.0~1.0): %.2f' % cpu_avg_eff)

def efficiency_monitor_start(data):
    gpu_efficiency_rating_over_time = []
    cpu_efficiency_rating_over_time = []
    while do_monitor:
        # monitor efficiency
        # TODO: also add to tensorboard
        cpu_loads = get_cpu_loads()
        gpu_loads, gpu_efficency_rating = get_gpu_loads_and_efficiency(GPU_IDS)

        # if rating is 0, gpu is probably just not in use (could be invalid data?? or bottlenecked on other area of app...might be important to know anyway)

        curtime = current_milli_time()

        cpu_efficiency = sum(cpu_loads) / float(len(cpu_loads))
        cpu_efficiency_rating_over_time.append((curtime, cpu_efficiency))

        #if efficency_rating != 0.0 and 0.0 not in gpu_loads:
        gpu_efficiency_rating_over_time.append((curtime, gpu_efficency_rating))
        if len(gpu_efficiency_rating_over_time) > 0 and len(gpu_efficiency_rating_over_time) % SHOW_EFFICIENCY_EVERY_N == 0:
            efficiency_show(gpu_efficiency_rating_over_time, cpu_efficiency_rating_over_time)

        # we don't want the measurement to get stuck on a certain point in the epoch
        # so sleep every random amount of seconds
        time.sleep(0.5+random.random())

joint_model = (args.joint_model_type != None)

#parser.add_argument('--joint-model-type', type=str, default=None,
#parser.add_argument('--joint-model-data-dir', type=str, default=None,
#                    help='Secondary data directory to load training data from (enables joint training)')

if joint_model:
    config2 = copy.deepcopy(config)
    config2.output_path = args.model_output_dir[:-1] + '_stage2/'

    #try:
    if not args.evaluate:
        os.mkdir(config2.output_path)
    #except:
    #pass

    config2.model_output = config2.output_path + 'output/'

    #try:
    if not args.evaluate:
        os.mkdir(config2.model_output)
    #except:
    #    pass

    config2.log_path = config2.model_output + 'log.txt'

    # have to load all as lexicon anyway, even during evaluate
    stage2_train = BIODataInput('data/%s/%s_train.txt' % (args.model_data_dir, args.joint_model_type))
    stage2_test = BIODataInput('data/%s/%s_test.txt' % (args.model_data_dir, args.joint_model_type))
    stage2_dev = BIODataInput('data/%s/%s_dev.txt' % (args.model_data_dir, args.joint_model_type))
    stage2_lexicon = BIOLexicon()
    stage2_lexicon.add_lexicon_data(stage2_train.get_lexicon())
    #stage2_lexicon.add_lexicon_data(stage2_test.get_lexicon())
    #stage2_lexicon.add_lexicon_data(stage2_dev.get_lexicon())

    ## FIXME: make sure to delete existing dir first
    stage2_lexicon.save('%slexicon' % config2.output_path)

    g1 = tf.Graph()
    with g1.as_default() as g:
        with tf.variable_scope(args.model_type):
            model = MorphingTaggingModel(lexicon=lexicon, config=config)

        with tf.variable_scope(args.joint_model_type):
            stage2_model = MorphingTaggingModel(lexicon=stage2_lexicon, config=config2)

        model.model_type = args.model_type
        stage2_model.model_type = args.joint_model_type

        joint_model = JointModel()
        joint_model.add_submodel(model)
        joint_model.add_submodel(stage2_model)

        if not args.evaluate:
            # begin efficiency monitoring
            do_monitor = True
            t = threading.Thread(target=efficiency_monitor_start, args=('',))
            t.start()
            joint_model.train(sent_train=(train.sentences, stage2_train.sentences), sent_dev=(dev.sentences, stage2_dev.sentences))
            do_monitor = False # stop monitoring

        ## TODO: joint inference not yet supported
        ## TODO add oov_out=args.eval_oov_tokens_file

        with tf.variable_scope(args.model_type, reuse=True):
            model.evaluate(sent_test=test.sentences, oov_out=args.eval_oov_tokens_file+'.morph')

        with tf.variable_scope(args.joint_model_type, reuse=True):
            stage2_model.evaluate(sent_test=stage2_test.sentences, oov_out=args.eval_oov_tokens_file+'.tag')

        '''
        if args.eval_inference_file != None:
            inference = BIODataInput(args.eval_inference_file)

            with tf.variable_scope(args.model_type, reuse=True):
                model.evaluate(sent_test=inference.sentences)

            with tf.variable_scope(args.joint_model_type, reuse=True):
                stage2_model.evaluate(sent_test=stage2_inference.sentences)
        else:
            with tf.variable_scope(args.model_type, reuse=True):
                model.evaluate(sent_test=test.sentences)

            with tf.variable_scope(args.joint_model_type, reuse=True):
                stage2_model.evaluate(sent_test=stage2_test.sentences)
        '''
else:
    g1 = tf.Graph()
    with g1.as_default() as g:
        with tf.variable_scope(args.model_type):
            model = MorphingTaggingModel(lexicon=lexicon, config=config)

            if not args.evaluate:
                # begin efficiency monitoring
                do_monitor = True
                t = threading.Thread(target=efficiency_monitor_start, args=('',))
                t.start()
                model.train(sent_train=train.sentences, sent_dev=dev.sentences)
                do_monitor = False # stop monitoring

            if args.eval_inference_file != None:
                inference = BIODataInput(args.eval_inference_file)
                model.evaluate(sent_test=inference.sentences, inference_only=True, oov_out=args.eval_oov_tokens_file)
            else:
                model.evaluate(sent_test=test.sentences, oov_out=args.eval_oov_tokens_file)
