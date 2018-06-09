# import as child logger
import logging
logger = logging.getLogger('BIOTools')

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
    Get inputs labels, chunked by (idx, val)
    Multiple inputs are concatenated by '-': TODO maybe better way
    val contains inputs + '/' + label

    Makes evaluation easier
    '''
    def get_input_and_label_chunks(self):
        chunk_set = set()

        assert len(self.inputs) == len(self.labels)

        for idx, label in enumerate(self.labels):
            chunk_set.add((idx, '-'.join(self.inputs[idx]) + '/' + label))

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
