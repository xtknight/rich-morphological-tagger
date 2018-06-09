import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : [%(name)s] : %(message)s', level=logging.INFO)
logger = logging.getLogger('DatasetStatistics')

parser = argparse.ArgumentParser(description='Print statistics about a dataset and discover duplicate data.')
# Required positional argument

parser.add_argument('model_type', type=str,
                    help='morph or tag')
parser.add_argument('model_data_dir', type=str,
                    help='Directory to load data from (data/model_data_dir/model_type_{train,dev,test}.txt')

args = parser.parse_args()


class BIODataSentence(object):
    def __init__(self):
        # array of inputs for this 'word' in the sentence
        self.inputs = []

        # label for this 'word' in the sentence (only one label possible)
        self.labels = []

        # cached sentence length
        self.sentence_length = 0

    def is_equal(self, other_sent):
        if self.sentence_length != other_sent.sentence_length:
            return False

        if self.inputs != other_sent.inputs:
            return False

        if self.labels != other_sent.labels:
            return False

        return True

    '''
    Not perfect, but helps optimize when finding duplicates
    '''
    def __hash__(self):
        return hash(self.sentence_length) * hash(tuple(self.inputs)) * hash(tuple(self.labels))

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
                    retval.append(embed_id_dict[embed_oov_token])
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
                    retval.append(embed_id_dict[embed_oov_token])
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
                    #assert input_unit not in BIOLexicon.ReservedInputTokens, '%s: wordline at para idx %d, wordline idx %d: input unit must not be reserved token' % (fn, pidx, widx)
                    word_inputs.append(input_unit)

                if self.num_inputs != None:
                    assert len(word_inputs) == self.num_inputs, '%s: wordline at para idx %d, wordline idx %d: number of inputs differs per sentence' % (fn, pidx, widx)
                else:
                    self.num_inputs = len(word_inputs)
                    logger.info('%s: detected %d input(s) and 1 label' % (fn, self.num_inputs))

                sent.inputs.append(tuple(word_inputs))

                #assert all_ins[-1] not in BIOLexicon.ReservedLabelTokens, '%s: wordline at para idx %d, wordline idx %d: label must not be reserved token' % (fn, pidx, widx)
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


train = BIODataInput('data/%s/%s_train.txt' % (args.model_data_dir, args.model_type))
test = BIODataInput('data/%s/%s_test.txt' % (args.model_data_dir, args.model_type))
dev = BIODataInput('data/%s/%s_dev.txt' % (args.model_data_dir, args.model_type))

dupes = dict()

setidx_to_str = {0: 'train', 1: 'test', 2: 'dev'}
setidx_to_obj = {0: train, 1: test, 2: dev}

overlap_counts = dict()

for s1 in setidx_to_str.values():
    for s2 in setidx_to_str.values():
        # sorted prevents duplicates <train, dev>, <dev, train>
        overlap_counts['<%s>' % ', '.join(sorted([s1, s2]))] = 0

#for idx, train_sent in enumerate(train.sentences):
#    dupes[hash(train_sent)] = (0, idx) # training set

# dupes can happen in training set too
for idx, train_sent in enumerate(train.sentences):
    h = hash(train_sent)
    if h in dupes:
        dset, didx = dupes[h]
        o = setidx_to_obj[dset]
        if o.sentences[didx].is_equal(train_sent):
            print('Duplicate data point <%s, train>: %s' % (setidx_to_str[dset], train_sent))
            overlap_counts['<%s>' % ', '.join(sorted(['train', setidx_to_str[dset]]))] += 1
        else:
            print('Hash collision', h)
    else:
        dupes[h] = (0, idx) # training set

for idx, test_sent in enumerate(test.sentences):
    h = hash(test_sent)
    if h in dupes:
        dset, didx = dupes[h]
        o = setidx_to_obj[dset]
        if o.sentences[didx].is_equal(test_sent):
            print('Duplicate data point <%s, test>: %s' % (setidx_to_str[dset], test_sent))
            overlap_counts['<%s>' % ', '.join(sorted(['test', setidx_to_str[dset]]))] += 1
        else:
            print('Hash collision', h)
    else:
        dupes[h] = (1, idx) # testing set

for idx, dev_sent in enumerate(dev.sentences):
    h = hash(dev_sent)
    if h in dupes:
        dset, didx = dupes[h]
        o = setidx_to_obj[dset]
        if o.sentences[didx].is_equal(dev_sent):
            print('Duplicate data point <%s, dev>: %s' % (setidx_to_str[dset], dev_sent))
            overlap_counts['<%s>' % ', '.join(sorted(['dev', setidx_to_str[dset]]))] += 1
        else:
            print('Hash collision', h)
    else:
        dupes[h] = (2, idx) # dev set

print(overlap_counts)

'''
for train_sent in train.sentences:
    for test_sent in test.sentences:
        if train_sent.is_equal(test_sent):
            print('Duplicate data point <train, test>:', train_sent)
    for dev_sent in dev.sentences:
        if train_sent.is_equal(dev_sent):
            print('Duplicate data point <train, dev>:', train_sent)

for test_sent in test.sentences:
    for dev_sent in dev.sentences:
        if test_sent.is_equal(dev_sent):
            print('Duplicate data point <test, dev>:', test_sent)
'''
