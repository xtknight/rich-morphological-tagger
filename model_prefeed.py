import os
import sys
import copy
import logging
import numpy as np
import tensorflow as tf
from general_utils import minibatches, Progbar

# amatteson@blp-deep-server-1:~/newunitag$ LD_PRELOAD=/home/amatteson/cudnn-8.0-6.0/cuda/lib64/libcudnn.so.6 python3 model.py 
# TODO: try limiting sentence size to max size per-batch (could be much more efficient)
# TODO: add pre-batching and ASGD multi-GPU training
# TODO: add joint training of two different models (morphing and tagging)

# maximum input units (pad until this number; disallow longer input)
MAX_UNIT_COUNT = 400

logging.basicConfig(format='%(asctime)s : %(levelname)s : [%(name)s] : %(message)s', level=logging.INFO)
logger = logging.getLogger('Model')

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

class BIOLexicon(object):
    UNK_TOK = '<UNK>'
    PAD_TOK = '<PAD>'
    ReservedInputTokens = [UNK_TOK, PAD_TOK]
    ReservedLabelTokens = [UNK_TOK, PAD_TOK]

    def __init__(self):
        self.sets = None

    '''
    Concatenate existing lexicon with more lexicon data

    (Useful when reading train, test, dev files separately and adding each one's
    lexicon)
    '''
    def add_lexicon_data(self, other_sets):
        assert len(other_sets) >= 2, 'must have at least one input and one label'

        if self.sets != None:
            assert len(other_sets) == len(self.sets), 'lexicon sets must be same size'
        else:
            self.sets = []
            for i in range(len(other_sets)):
                self.sets.append(set())

        # union new set with current set
        for i in range(len(other_sets)):
            self.sets[i] = self.sets[i] | other_sets[i]

    '''
    Get the corresponding lexicon (vocabulary) for input N
    '''
    def get_input_lexicon(self, input_idx):
        # make sure there are enough sets as far as inputs and the extra label set (-1)
        assert self.sets != None and len(self.sets) >= 2 and len(self.sets)-1 > input_idx
        return self.sets[input_idx]

    '''
    Get the corresponding lexicon (vocabulary) for the label
    '''
    def get_label_lexicon(self):
        # make sure there is at least one input and one label
        assert self.sets != None and len(self.sets) >= 2
        return self.sets[-1]

    '''
    Get the corresponding ordered lexicon (vocabulary) for input N

    This is the lexicon with a consistent order and with special UNK tokens at
    the beginning (so that the reserved tokens always map to index 0,1,2,etc...)

    Returns forward and reverse lookup dictionaries (i->item, item->i)
    '''
    def get_input_lexicon_for_training(self, input_idx):
        ## this assumes that the input lexicon contains strings
        input_lex = []
        for r in BIOLexicon.ReservedInputTokens:
            input_lex.append(r)
        input_lex += sorted(self.get_input_lexicon(input_idx))

        fwd_dict = {}
        rev_dict = {}
        for i, item in enumerate(input_lex):
            assert i not in fwd_dict
            fwd_dict[i] = item

            assert item not in rev_dict
            rev_dict[item] = i

        return fwd_dict, rev_dict

    '''
    Get the corresponding ordered lexicon (vocabulary) for the label

    This is the lexicon with a consistent order and with special UNK tokens at
    the beginning (so that the reserved tokens always map to index 0,1,2,etc...)

    Returns forward and reverse lookup dictionaries (i->item, item->i)
    '''
    def get_label_lexicon_for_training(self):
        ## this assumes that the label lexicon contains strings
        label_lex = []
        for r in BIOLexicon.ReservedLabelTokens:
            label_lex.append(r)
        label_lex += sorted(self.get_label_lexicon())

        fwd_dict = {}
        rev_dict = {}
        for i, item in enumerate(label_lex):
            assert i not in fwd_dict
            fwd_dict[i] = item

            assert item not in rev_dict
            rev_dict[item] = i

        return fwd_dict, rev_dict

    def __str__(self):
        return 'BIOLexicon{' + ', '.join(['Input%d: %d items' % (i, len(self.get_input_lexicon(i))) for i in range(len(self.sets)-1)]) + ', Label: %d items' % len(self.get_label_lexicon()) + '}'

    def __repr__(self):
        return self.__str__()

    '''
    Save all lexicon sets (max 999)

    NOTE: assumes that ReservedInputTokens and ReservedLabelTokens stayed the same,
          and that the sorted() function operates the same each time

    FIXME: make sure that when saving, other items past the current number of sets
           don't exist
    '''
    def save(self, fn_prefix):
        for i in range(len(self.sets)):
            with open(fn_prefix + '_%03d.lex' % i, 'w', encoding='utf-8') as fd:
                for idx, item in enumerate(self.sets[i]):
                    assert '\n' not in item
                    assert len(item) > 0
                    if idx < len(self.sets[i])-1:
                        fd.write('%s\n' % item)
                    else: # skip extra newline on last item
                        fd.write('%s' % item)

    '''
    Load all lexicon sets (min filename: 000, max filename: 999)

    NOTE: assumes that ReservedInputTokens and ReservedLabelTokens stayed the same,
          and that the sorted() function operates the same each time
    '''
    def load(self, fn_prefix):
        self.sets = []

        # detect number of sets that exist
        for i in range(1000):
            try:
                with open(fn_prefix + '_%03d.lex' % i, 'r', encoding='utf-8') as fd:
                    self.sets.append(set())
            except FileNotFoundError:
                break

        assert len(self.sets) >= 2, 'must have at least one input and one label'

        for i in range(len(self.sets)):
            with open(fn_prefix + '_%03d.lex' % i, 'r', encoding='utf-8') as fd:
                for ln in fd.read().split('\n'):
                    assert len(ln) > 0, 'empty input line in lexicon'

                    # make sure reserved input tokens are not in the set
                    if i < len(self.sets)-1:
                        assert ln not in BIOLexicon.ReservedInputTokens
                    else:
                        assert ln not in BIOLexicon.ReservedLabelTokens

                    self.sets[i].add(ln)

class BIODataSentence(object):
    def __init__(self):
        # array of inputs for this 'word' in the sentence
        self.inputs = []

        # label for this 'word' in the sentence (only one label possible)
        self.labels = []

        # cached sentence length
        self.sentence_length = 0

    '''
    Get inputs for specified index, but padded

    If embed_id_dict is not None, use as dictionary to embed each input item
    '''
    def get_inputs_padded(self, input_idx, max_length, padding_token, embed_id_dict=None, embed_oov_token=None):
        assert len(self.inputs) > 0 # so that checking len(self.inputs[0]) is possible
        assert input_idx >= 0 and input_idx < len(self.inputs[0])
        assert type(max_length) is int

        # this assertion may fail during the evaluation phase
        # FIXME: how to handle a sentence that is too long for the input?
        assert len(self.inputs) <= max_length, '%d > %d' % (len(self.inputs), max_length)

        retval = []
        for inp in self.inputs:
            if embed_id_dict != None:
                if inp[input_idx] in embed_id_dict:
                    retval.append(embed_id_dict[inp[input_idx]])
                else:
                    assert embed_oov_token != None, 'must specify OOV token for embedding'
                    retval.append(embed_oov_token)
            else:
                retval.append(inp[input_idx])

        if max_length > len(retval):
            # add padding
            for i in range(max_length - len(retval)):
                if embed_id_dict != None:
                    # embed padding
                    retval.append(embed_id_dict[padding_token])
                else:
                    retval.append(padding_token)

        return retval

    '''
    Get labels, but padded

    If embed_id_dict is not None, use as dictionary to embed each input item
    '''
    def get_labels_padded(self, max_length, padding_token, embed_id_dict=None, embed_oov_token=None):
        assert type(max_length) is int
        assert len(self.labels) <= max_length

        retval = []
        for lab in self.labels:
            if embed_id_dict != None:
                if lab in embed_id_dict:
                    retval.append(embed_id_dict[lab])
                else:
                    assert embed_oov_token != None, 'must specify OOV token for embedding'
                    retval.append(embed_oov_token)
            else:
                retval.append(lab)

        if max_length > len(retval):
            # add padding
            for i in range(max_length - len(retval)):
                if embed_id_dict != None:
                    # embed padding
                    retval.append(embed_id_dict[padding_token])
                else:
                    retval.append(padding_token)

        return retval

    '''
    Get labels, chunked by (idx, val)

    Makes evaluation easier
    '''
    def get_label_chunks(self):
        chunk_set = set()

        for idx, label in enumerate(self.labels):
            chunk_set.add((idx, label))

        return chunk_set

    '''
    Concatenate all inputs and labels by tab and return string
    '''
    def __str__(self):
        return '\n'.join(['\t'.join(list(inputline[0]) + [inputline[1]]) for inputline in zip(self.inputs, self.labels)])

    def __repr__(self):
        return self.__str__()

class BIODataInput(object):
    def __init__(self, fn):
        self.fn = fn
        self.sentences = []
        self.num_inputs = None

        blocks = None
        min_sent_length = float('inf')
        max_sent_length = float('-inf')

        with open(fn, 'r', encoding='utf-8') as fd:
            blocks = fd.read().split('\n\n')

        assert len(blocks) > 0, 'input is empty'

        for pidx, item in enumerate(blocks):
            sent = BIODataSentence()
            item = item.strip()
            if len(item) == 0:
                logger.error('%s: empty block at idx %d' % (fn, pidx))
                continue
            for widx, wordline in enumerate(item.split('\n')):
                wordline = wordline.strip()
                if len(wordline) == 0:
                    logger.error('%s: empty wordline at para idx %d, wordline idx %d' % (fn, pidx, widx))
                    continue
                all_ins = wordline.split('\t')
                assert len(all_ins) >= 2, '%s: wordline at para idx %d, wordline idx %d: must have at least one input and one label' % (fn, pidx, widx)
                word_inputs = []
                for input_unit in all_ins[:-1]:
                    if not input_unit:
                        continue
                    assert input_unit not in BIOLexicon.ReservedInputTokens, '%s: wordline at para idx %d, wordline idx %d: input unit must not be reserved token' % (fn, pidx, widx)
                    word_inputs.append(input_unit)

                if self.num_inputs != None:
                    assert len(word_inputs) == self.num_inputs, '%s: wordline at para idx %d, wordline idx %d: number of inputs differs per sentence' % (fn, pidx, widx)
                else:
                    self.num_inputs = len(word_inputs)
                    logger.info('%s: detected %d input(s) and 1 label' % (fn, self.num_inputs))

                sent.inputs.append(tuple(word_inputs))

                assert all_ins[-1] not in BIOLexicon.ReservedLabelTokens, '%s: wordline at para idx %d, wordline idx %d: label must not be reserved token' % (fn, pidx, widx)
                sent.labels.append(all_ins[-1])

            assert len(sent.inputs) == len(sent.labels), 'number of wordline inputs and labels must be same'
            sent.sentence_length = len(sent.labels)
            assert sent.sentence_length > 0, 'sentence length should be greater than zero'

            if sent.sentence_length < min_sent_length:
                min_sent_length = sent.sentence_length
            if sent.sentence_length > max_sent_length:
                max_sent_length = sent.sentence_length

            self.sentences.append(sent)

        logger.info('%s: read %d BIO sentences (min length: %d, max length: %d)' % (fn, len(self.sentences), min_sent_length, max_sent_length))

    '''
    Return set of inputs and labels that occur

    [set(input1), set(input2), set(inputN), set(label)]
    '''
    def get_lexicon(self):
        sets = []
        for i in range(self.num_inputs):
            sets.append(set())

        # for labels
        sets.append(set())

        for sidx, sent in enumerate(self.sentences):
            # add all inputs for each wordline to the corresponding set for that input
            for i in range(self.num_inputs):
                for widx, perword_inputs in enumerate(sent.inputs):
                    sets[i].add(perword_inputs[i])

            # add each wordline's label to the set
            for widx, word_label in enumerate(sent.labels):
                sets[-1].add(word_label)

        return sets

# TODO: joint model

'''
class JointModel(object):
    def __init__(self):
        self.models = []

    def add_submodel(self, submodel):
        self.models.append(submodel)

    def add_joint_loss_op(self):
        self.joint_loss = tf.sum(self.
'''

class MorphingTaggingModel(object):
    def __init__(self, lexicon, config):
        #nested_d = tf.constant(10.0, name="d")
        #assert nested_d.op.name == "morphing/nested/d", nested_d.op.name

        self.logger = logging.getLogger('MorphingTaggingModel')
        self.lexicon = lexicon
        self.config = config
        self.idx_to_unit0, self.unit0_to_idx = self.lexicon.get_input_lexicon_for_training(0)
        self.idx_to_label, self.label_to_idx = self.lexicon.get_label_lexicon_for_training()
    
        self.add_placeholders()
        self.add_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

        with tf.Session() as sess:
            sess.run(self.init)
            tvars = tf.trainable_variables()
            tvars_vals = sess.run(tvars)

            for var, val in zip(tvars, tvars_vals):
                print(var.name) # , val)  # Prints the name of the variable alongside its value.


    def add_placeholders(self):
        '''
        Adds placeholders to self
        '''
        # words or morphemes
        # shape = (batch size, max length of sentence in batch)
        self.unit0_ids = tf.placeholder(tf.int32, shape=[None, None], 
                        name='unit0_ids')

        #assert self.unit0_ids.op.name == "morphing/nested/unit0_ids", self.unit0_ids.op.name

        # number of "units" (words or morphemes)
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name='sequence_lengths')

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name='labels')

        # hyper parameters
        # dropout keep probability (1.0 means keep all data)
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], 
                        name='dropout_keep_prob')

        # learning rate
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], 
                        name='learning_rate')


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

            _unit0_embeddings = tf.Variable(initial_value=unit0_initial_value,
                                           name='_unit0_embeddings', dtype=tf.float32,
                                           trainable=self.config.do_unit_embedding_training[0])

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

    def add_logits_op(self):
        '''
        Adds logits to self
        '''

        # number of output labels
        nlabels = len(self.idx_to_label)

        with tf.variable_scope('bi-lstm'):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.combined_hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, 
                lstm_cell, self.combined_embeddings, sequence_length=self.sequence_lengths, 
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout_keep_prob)

        with tf.variable_scope('proj'):
            W = tf.get_variable('W', shape=[2*self.config.combined_hidden_size, nlabels],
                dtype=tf.float32, initializer=tf.random_normal_initializer( \
                    stddev=1.0 / (2*self.config.combined_hidden_size)**.5, \
                    seed=0))
            print('W.op.name', W.op.name)

            b = tf.get_variable('b', shape=[nlabels], dtype=tf.float32, 
                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.combined_hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, nlabels])


    def add_pred_op(self):
        '''
        Adds labels_pred to self
        '''
        if not self.config.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def add_loss_op(self):
        '''
        Adds loss to self
        '''
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            print('losses op name', losses.op.name)
            self.loss = tf.reduce_mean(losses)

        print('loss op name', self.loss.op.name)

        # for tensorboard
        tf.summary.scalar('loss', self.loss)


    def add_train_op(self):
        '''
        Add train_op to self
        '''
        current_scope = tf.get_default_graph().get_name_scope()
        print('current scope', current_scope)

        current_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 current_scope)
        print('current_train_vars', current_train_vars)

        with tf.variable_scope('train_step'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, var_list=current_train_vars)


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

        for sidx, sent in enumerate(bio_data_sentence_batch): # sent:BIODataSentence
            ## for now, only using first input
            input_batch.append(sent.get_inputs_padded(
                               input_idx=0,
                               max_length=MAX_UNIT_COUNT,
                               padding_token=BIOLexicon.PAD_TOK,
                               embed_id_dict=self.unit0_to_idx,
                               embed_oov_token=BIOLexicon.UNK_TOK))
            labels_batch.append(sent.get_labels_padded(
                               max_length=MAX_UNIT_COUNT,
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

        if self.dropout_keep_prob != None:
            feed[self.dropout_keep_prob] = dropout_keep_prob

        if self.learning_rate != None:
            feed[self.learning_rate] = learning_rate

        return feed, sequence_lengths


    '''
    Prepare a feed dict for the specified model based on the given input batch
    Fills tf variables and placeholders with necessary values for sentence batch

    bio_data_sentence_list: list of BIODataSentence objects
    dropout_keep_prob:      inverse dropout rate: 1.0 means keep all data
    learning_rate:          optimizer learning rate
    '''
    def prepare_feed_dict_optimized(self, bio_data_sentence_batch, dropout_keep_prob=None, learning_rate=None):
        input_batch = []
        labels_batch = []
        sequence_lengths = []

        for sidx, sent in enumerate(bio_data_sentence_batch): # sent:BIODataSentence
            ## for now, only using first input
            input_batch.append(sent.cache_inputs0_padded)
            labels_batch.append(sent.cache_labels_padded)
            sequence_lengths.append(sent.sentence_length)

            self.logger.debug('input_batch[%d] (len=%d): %s' % (sidx, len(input_batch[sidx]), str(input_batch[sidx])))
            self.logger.debug('labels_batch[%d] (len=%d): %s' % (sidx, len(labels_batch[sidx]), str(labels_batch[sidx])))
            self.logger.debug('sequence_length[%d]: %d' % (sidx, sequence_lengths[sidx]))

            assert len(input_batch[sidx]) == len(labels_batch[sidx])

        feed = {}
        feed[self.unit0_ids] = input_batch
        feed[self.sequence_lengths] = sequence_lengths
        feed[self.labels] = labels_batch

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

        prog = Progbar(target=nbatches)
        for i, sent_batch in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.prepare_feed_dict_optimized( \
                        bio_data_sentence_batch=sent_batch, \
                        dropout_keep_prob=self.config.dropout_keep_prob, \
                        learning_rate=self.config.learning_rate)

            #_, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            _, train_loss = sess.run([self.train_op, self.loss], feed_dict=fd)

            prog.update(i + 1, [('train loss', train_loss)])

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

        # dropout_keep_prob forced to 1.0 at inference time
        fd, sequence_lengths = self.prepare_feed_dict( \
                                   bio_data_sentence_batch=sents, \
                                   dropout_keep_prob=1.0)
        assert len(sequence_lengths) == len(sents)

        pred_sents = copy.deepcopy(sents)

        for sidx, s in enumerate(pred_sents):
        #    s.labels = []
            assert sequence_lengths[sidx] == len(s.inputs)
        #    for i in range(sequence_lengths[sidx]):
        #        s.labels.append(BIOLexicon.PAD_TOK)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params], 
                    feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                logit, transition_params)
                viterbi_sequences.append(viterbi_sequence)
            
            for sidx, s in enumerate(pred_sents):
                s.labels = []
                for i in range(sequence_lengths[sidx]):
                    ## FIXME: is it possible for the NN to return a non-sensible value here?
                    s.labels.append(self.idx_to_label[viterbi_sequences[sidx][i]])

            #return viterbi_sequences, sequence_lengths

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)

            ## TODO: check sequence_lengths against input lengths

            for sidx, s in enumerate(pred_sents):
                s.labels = []
                for i in range(sequence_lengths[sidx]):
                    ## FIXME: is it possible for the NN to return a non-sensible value here?
                    s.labels.append(self.idx_to_label[labels_pred[sidx][i]])

            #return labels_pred, sequence_lengths

        return pred_sents


    def run_evaluate(self, sess, test):
        '''
        Evaluates performance on specified test/dev set

        Args:
            sess: tensorflow session
            test: large BIODataSentence list (dev/test set)
        '''

        #nbatches = len(test) // self.config.batch_size
        #if len(test) % self.config.batch_size != 0:
        #    nbatches += 1

        correct_preds = 0
        total_preds = 0
        total_correct = 0
        correct_mod = 0
        total_mod = 0
        accs = []

        #prog = Progbar(target=nbatches)
        for i, gold_sent_batch in enumerate(minibatches(test, self.config.batch_size)):
            pred_sent_batch = self.predict_batch(sess, gold_sent_batch)

            assert len(gold_sent_batch) == len(pred_sent_batch)

            for sidx in range(len(gold_sent_batch)):
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

        self.logger.info('correct_preds: ' + str(correct_preds))
        self.logger.info('total_preds: ' + str(total_preds))
        self.logger.info('total_correct: ' + str(total_correct))

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        mod_p = correct_mod / total_mod if correct_mod > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1, mod_p


    def train(self, sent_train, sent_dev):
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True

        #with tf.Session(config=config) as sess:
        with tf.Session() as sess:
            sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info('Epoch {:} out of {:}'.format(epoch + 1, self.config.nepochs))

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
        
    def evaluate(self, sent_test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info('Testing model over test set')
            saver.restore(sess, self.config.model_output)
            acc, f1, mod_p = self.run_evaluate(sess, sent_test)
            self.logger.info('- test acc {:04.2f} - f1 {:04.2f} - mod prec {:04.2f}'.format(100*acc, 100*f1, 100*mod_p))

morph_train = BIODataInput('data/entire_500actions_400len/morph_train.txt')
morph_test = BIODataInput('data/entire_500actions_400len/morph_test.txt')
morph_dev = BIODataInput('data/entire_500actions_400len/morph_dev.txt')
morph_lexicon = BIOLexicon()
morph_lexicon.add_lexicon_data(morph_train.get_lexicon())
#morph_lexicon.add_lexicon_data(morph_test.get_lexicon())
#morph_lexicon.add_lexicon_data(morph_dev.get_lexicon())
print('morph lexicon', morph_lexicon)

morph_lexicon.save('morph_lexicon')

tag_train = BIODataInput('data/entire_500actions_400len/tag_train.txt')
tag_test = BIODataInput('data/entire_500actions_400len/tag_test.txt')
tag_dev = BIODataInput('data/entire_500actions_400len/tag_dev.txt')
tag_lexicon = BIOLexicon()
tag_lexicon.add_lexicon_data(tag_train.get_lexicon())
#tag_lexicon.add_lexicon_data(tag_test.get_lexicon())
#tag_lexicon.add_lexicon_data(tag_dev.get_lexicon())
print('tagging lexicon', tag_lexicon)

config = ModelConfig()
config.input_unit_embedding_sizes = [300]
config.do_unit_embedding_training = [True]
config.dropout_keep_prob = 0.7
config.combined_hidden_size = 300
config.nepochs = 5
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

config.crf = True

config.output_path = 'model/'
config.model_output = config.output_path + 'output/'
config.log_path = config.model_output + 'log.txt'
config.batch_size = 16

g1 = tf.Graph()
with g1.as_default() as g:
    with tf.variable_scope('morphing'):
        morph_model = MorphingTaggingModel(lexicon=morph_lexicon, config=config)

        sentcount = len(morph_train.sentences)

        morph_train.cached_sentences = []

        for sidx, sent in enumerate(morph_train.sentences):
            if sidx % 20 == 0:
                #print('Preprocesing sentence %d/%d' % (sidx, sentcount))
                pass

            if len(sent.labels) > MAX_UNIT_COUNT:
                #print('Skipping sentence: too long')
                continue

            sent.cache_inputs0_padded = sent.get_inputs_padded(
                                       input_idx=0,
                                       max_length=MAX_UNIT_COUNT,
                                       padding_token=BIOLexicon.PAD_TOK,
                                       embed_id_dict=morph_model.unit0_to_idx,
                                       embed_oov_token=BIOLexicon.UNK_TOK)

            sent.cache_labels_padded = sent.get_labels_padded(
                                       max_length=MAX_UNIT_COUNT,
                                       padding_token=BIOLexicon.PAD_TOK,
                                       embed_id_dict=morph_model.label_to_idx,
                                       embed_oov_token=BIOLexicon.UNK_TOK)

            morph_train.cached_sentences.append(sent)

        morph_model.train(sent_train=morph_train.cached_sentences, sent_dev=morph_dev.sentences)
        morph_model.evaluate(sent_test=morph_test.sentences)

'''
#g2 = tf.Graph()
with g1.as_default() as g:
    with tf.variable_scope('tagging'):
        # HACK: debugging only: use all dev set
        tag_model = MorphingTaggingModel(lexicon=tag_lexicon, config=config)
        tag_model.train(sent_train=tag_dev.sentences, sent_dev=tag_dev.sentences)
    #    tag_model.evaluate(sent_test=tag_dev.sentences)


tf.reset_default_graph()
'''

#joint_loss = morph_model.loss + tag_model.loss
