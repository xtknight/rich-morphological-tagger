import sys
import unicodedata
import os
import random
import re
import argparse
import logging

from conll_utils import *
from korean_morph_seq2 import *
from ud_converter import simple_ngram_oracle

'''
Helps prepare data to make it usable
with main.py

TODO: would be nice to make ignore-training-corpus-sent-ids bidirectional
 so that stuff evicted from tag set can also be evicted from morph set
 however, currently for tagging we just include everything so it doesn't matter
 (and unit count only gets smaller during tagging, but if unit count got bigger
  we could run into issues)

  Clarify that eviction refers to sentences either removed due to errors
  or through action filtering

  We also have eviction that lands sentences in all_sentences and later
  action-based eviction but it uses the same member variable.
  Clarify this as well. This is useful so that blatantly erroneous sentences
  are ignored before any cross-validation takes place to keep as much balance
  as possible.

  After cross-validation eviction still occurs, but only within the training
  set. We keep sentences with rare actions in test and dev.
'''

#
# Sample command line:
# python3 preprocess_data.py --mode 1 /mnt/deeplearn/CoNLL_2017/UD_Russian/ru-ud-all.conllu --output-suffix=ru --no-shuffle --max-actions-training=100
# python3 preprocess_data.py --mode 0 /home/andy/dev/ash-morpher/corpus/sejong-balanced-spoken.conllu --output-suffix=sejong --no-shuffle --lemma-morphemes --debug --max-actions-training=700
# python3 preprocess_data.py --mode 2   /home/andy/dev/ash-morpher/corpus/sejong-balanced-spoken.conllu --output-suffix=sejong --no-shuffle --lemma-morphemes --enrich-with-original-word --debug
# python3 preprocess_data.py --mode 0 --lemma-as-stem --output-breaklevel-flags --no-shuffle /mnt/deeplearn/CoNLL_2017/UD_Japanese/ja-ud-all.conllu --debug --output-suffix ja
# python3 preprocess_data.py --no-shuffle --mode 1 --use-lemma-morphemes-for-morphtag --lemma-morphemes --output-suffix ko-comb-morph sejong-practice-100000.conllu 

# train,dev,test are already split at the CoNLL level, then this program
# can also be run with --training-split 100 --no-shuffle --output-suffix={ud-train,ud-dev,ud-test}
# for ud train,dev,test conllu files separately
parser = argparse.ArgumentParser(description='Process CoNLL-U corpus and output training sets for morphological analysis.')

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input CoNLL-U file (Universal Dependencies)')
parser.add_argument('--mode', type=int, default=0,
                    help='Processing mode (default 0). 0: unit-to-unit transform (morpheme segmentation/transformation/stem identification using oracle over chars or char n-grams); 1: word-to-morphtag (per-word morphological tagging; does not require mode 0 first); 2: morph-to-morphtag (per-morpheme tagging with optional enriched original word information; requires mode 0 (with same u2u-oracle mode) first; word-level UPOSTAG is ignored)')
parser.add_argument('--u2u-oracle', type=int, default=0,
                    help='Oracle for unit2unit mode (default 0). 0: smart char2char transform with action optimizer; 1: language-tuned char n-gram transformer')
parser.add_argument('--no-posmorph', action='store_true', default=False,
                    help='For mode 1, don\'t preface the UPOSTAG to the word tag action')
parser.add_argument('--config-name', type=str, default='ud',
                    help='Name of configuration to output to configs/ (default \'ud\'). Output filename: language code is probably a good choice.')
parser.add_argument('--output-format', type=str, default='seq2seq',
                    help='Output format (default seq2seq). seq2seq, MarMoT')
parser.add_argument('--enrich-with-original-word', action='store_true', default=False,
                    help='Optional output column 2: for mode 2, enrich the '
                         'per-morpheme tagger with the original per-transformation word by adding an extra input column')
parser.add_argument('--output-breaklevel-flags', action='store_true', default=False,
                    help='Optional output column 3: for mode 0, output input '
                         'flags (auxiliary input to seq2seq to indicate '
                         'whitespace/break status of word at the character level)')
parser.add_argument('--training-split', type=int, default=80,
                    help='Training split (default 80). Percentage of data to be left for training. If 100, cross-validation is disabled and full file is output. Useful if your input is already split, like UD.')
parser.add_argument('--testing-split', type=int, default=10,
                    help='Testing split (default 10). Percentage of data to be left for testing')
parser.add_argument('--dev-split', type=int, default=10,
                    help='Dev split (default 10). Percentage of data to be left for dev')
parser.add_argument('--max-actions-training', type=int, default=0,
                    help='Maximum number of actions to leave in the training set (default 0, unlimited). Results in all sentences that contain words or characters using uncommon actions being removed. Reasonable values are probably 200-2000 for RNN memory constraints. It\'s probably bad practice to remove from testing and dev sets because we can\'t accurately measure the performance impact of the limited training action set and \'OOV\' actions.')
parser.add_argument('--no-shuffle', action='store_true', default=False,
                    help='Don\'t perform a random shuffle for all sentences. Makes output deterministic.')
parser.add_argument('--min-word-count', type=int, default=0,
                    help='Ignore all sentences with less words than this (default 0: no limit).')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Enable verbose debug lines')
parser.add_argument('--lemma-morphemes', action='store_true', default=False,
                    help='Enable if CoNLL-U lemma field contains a combination of morphemes (e.g., Korean Sejong Corpus)')
parser.add_argument('--lemma-as-stem', action='store_true', default=False,
                    help='Enable if CoNLL-U lemma field contains the word stem (e.g., Universal Dependencies Corpus)')
parser.add_argument('--use-lemma-morphemes-for-morphtag', action='store_true', default=False,
                    help='Combine lemme morphemes and use them as the morph tag (for mode 1)')
parser.add_argument('--spmrl', action='store_true', default=False,
                    help='Input is SPMRL corpus (morphemes in LEMMA and '
                         'UPOSTAG fields separated by plus sign)')
parser.add_argument('--ud-multilemma', action='store_true', default=False,
                    help='Input is Universal Dependencies corpus (morphemes '
                    'specified as subtokens')
parser.add_argument('--write-config', action='store_true', default=False,
                    help='Write config and run build_data.py automatically on the config file')
parser.add_argument('--config-fasttext', type=str, default='',
                    help='Specify pre-trained word vector file for config file. If unspecified, pre-trained word vectors are not loaded')
parser.add_argument('--config-col1-embed-size', type=int, default=300,
                    help='Specify embedding size for column 1')
parser.add_argument('--config-col2-embed-size', type=int, default=300,
                    help='Specify embedding size for column 2 (if column 2 doesn\'t exist, size 1 is forced)')
parser.add_argument('--conll-jamo-split', action='store_true', default=False,
                    help='Split all Korean words into Jamo at the corpus level')
parser.add_argument('--append-underscore-to-word', action='store_true', default=False,
                    help='Append an underscore to each word to signify whitespace')
parser.add_argument('--max-char-count', type=int, default=400,
                    help='Ignore all sentences with more chars than this [including underscore whitespace tokens] (default: 400)')
parser.add_argument('--keep-duplicates', action='store_true', default=False,
                    help='Keep duplicate sentences in the corpus')
parser.add_argument('--output-evicted-training-sent-ids', type=str, default=None,
                    help='Output original corpus indices of sentences with evicted training actions to the specified file (useful for ensuring aligned training sets among multiple modes with different evicted actions')
parser.add_argument('--ignore-training-corpus-sent-ids', type=str, default=None,
                    help='Ignore sentence IDs specified in this file (based on original sentence IDs in corpus before any sentence are evicted). Useful for omitting sentences that were evicted in a previous mode. For consistency amongst modes, this happens AFTER non-shuffled identical cross-validation splitting takes place.')
#parser.add_argument('--max-input-sentences', type=int, default=0,
#                    help='Specifies the maximum number of input sentences to process from the CoNLL input')
#parser.add_argument('--add-orig-word', action='store_true', default=False,
#                    help='For mode 2, add original word output to each post-transform character for POS tagging')
#parser.add_argument('--rich-character-info', action='store_true', default=False,
#                    help='For mode 1, indicate that the language encodes rich information in each character (e.g., #Korean). Enables per-character output and extra break-level flags are automatically enabled.')


# used when generating filename
mode_to_str = {0: 'unit2unit', 1: 'word2tag', 2: 'morph2tag'}

args = parser.parse_args()
assert 0 <= args.mode <= 2 # others not supported yet

# make code easier to read later
args.posmorph = not args.no_posmorph
args.shuffle = not args.no_shuffle

# mutually exclusive options
if args.ud_multilemma:
    assert not args.spmrl
    assert not args.lemma_as_stem
    assert not args.lemma_morphemes

if args.output_evicted_training_sent_ids != None:
    assert not args.shuffle, 'Don\'t shuffle when using output_evicted_training_sent_ids: instead, pre-shuffle input file before running this script'
    assert args.ignore_training_corpus_sent_ids == None, 'Don\'t use both output sent ids and filter sent ids functions at once: ids may not match'
if args.ignore_training_corpus_sent_ids != None:
    assert not args.shuffle, 'Don\'t shuffle when using ignore_training_corpus_sent_ids: instead, pre-shuffle input file before running this script'
    assert args.output_evicted_training_sent_ids == None, 'Don\'t use both output sent ids and filter sent ids functions at once: ids may not match'


###### DEBUG ######
TODO_pieces = ['a', 'abil', 'acağ', 'acak', 'alım', 'ama', 'an', 'ar', 'arak', 'asın', 'asınız', 'ayım', 'da', 'dan', 'de', 'den', 'dı', 'dığ', 'dık', 'dıkça', 'dır', 'di', 'diğ', 'dik', 'dikçe', 'dir', 'du', 'duğ', 'duk', 'dukça', 'dur', 'dü', 'düğ', 'dük', 'dükçe', 'dür', 'e', 'ebil', 'eceğ', 'ecek', 'elim', 'eme', 'en', 'er', 'erek', 'esin', 'esiniz', 'eyim', 'ı', 'ım', 'ımız', 'ın', 'ınca', 'ınız', 'ıp', 'ır', 'ıyor', 'ız', 'i', 'im', 'imiz', 'in', 'ince', 'iniz', 'ip', 'ir', 'iyor', 'iz', 'k', 'ken', 'la', 'lar', 'ları', 'ların', 'le', 'ler', 'leri', 'lerin', 'm', 'ma', 'madan', 'mak', 'maksızın', 'makta', 'maktansa', 'malı', 'maz', 'me', 'meden', 'mek', 'meksizin', 'mekte', 'mektense', 'meli', 'mez', 'mı', 'mış', 'mız', 'mi', 'miş', 'miz', 'mu', 'muş', 'mü', 'muz', 'müş', 'müz', 'n', 'nın', 'nız', 'nin', 'niz', 'nun', 'nuz', 'nün', 'nüz', 'r', 'sa', 'se', 'sı', 'sın', 'sın', 'sınız', 'sınlar', 'si', 'sin', 'sin', 'siniz', 'sinler', 'su', 'sun', 'sun', 'sunlar', 'sunuz', 'sü', 'sün', 'sün', 'sünler', 'sünüz', 'ta', 'tan', 'te', 'ten', 'tı', 'tığ', 'tık', 'tıkça', 'tır', 'ti', 'tiğ', 'tik', 'tikçe', 'tir', 'tu', 'tuğ', 'tuk', 'tukça', 'tur', 'tü', 'tüğ', 'tük', 'tükçe', 'tür', 'u', 'um', 'umuz', 'un', 'un', 'unca', 'unuz', 'up', 'ur', 'uyor', 'uz', 'ü', 'üm', 'ümüz', 'ün', 'ün', 'ünce', 'ünüz', 'üp', 'ür', 'üyor', 'üz', 'ya', 'yabil', 'yacağ', 'yacak', 'yalım', 'yama', 'yan', 'yarak', 'yasın', 'yasınız', 'yayım', 'ydı', 'ydi', 'ydu', 'ydü', 'ye', 'yebil', 'yeceğ', 'yecek', 'yelim', 'yeme', 'yen', 'yerek', 'yesin', 'yesiniz', 'yeyim', 'yı', 'yım', 'yın', 'yınca', 'yınız', 'yıp', 'yız', 'yi', 'yim', 'yin', 'yince', 'yiniz', 'yip', 'yiz', 'yken', 'yla', 'yle', 'ymış', 'ymiş', 'ymuş', 'ymüş', 'ysa', 'yse', 'yu', 'yum', 'yun', 'yunca', 'yunuz', 'yup', 'yü', 'yuz', 'yüm', 'yün', 'yünce', 'yünüz', 'yüp', 'yüz']

# for consistency:
# sort by alphabetical order
TODO_pieces = sorted(TODO_pieces)
# then sort by longest first (stable sort, so alphabetical order should be maintained)
TODO_pieces = sorted(TODO_pieces, key=len, reverse=True)
###### DEBUG ######

###### FINDING DUPLICATE DATA POINTS ######
# see dataset_statistics.py
# duplicate sentences. key: hash, value: (set_num, idx_in_set)
dupes = dict()

setidx_to_str = {0: 'all'} #, 1: 'test', 2: 'dev'}
#setidx_to_obj = {0: train, 1: test, 2: dev}

# overlap counts for '<train, dev>', '<train, train>', etc
overlap_counts = dict()
for s1 in setidx_to_str.values():
    for s2 in setidx_to_str.values():
        # sorted prevents duplicates <train, dev>, <dev, train>
        overlap_counts['<%s>' % ', '.join(sorted([s1, s2]))] = 0

# however, in our case, we only find duplicates for one set
# (we split at the last second, so we don't know whether those data points
# would have gone into test or dev). besides, it's better than way
# so that we can get a clean split on clean data rather than leaving
# the data count unbalanced after removing invalid items
###### FINDING DUPLICATE DATA POINTS ######

UNIT_TO_UNIT = 0
WORD_TO_MORPHTAG = 1
MORPH_TO_MORPHTAG = 2

args.output_format = args.output_format.lower()
assert args.output_format in ['seq2seq', 'marmot']

assert args.min_word_count >= 0
assert args.max_actions_training >= 0

if args.lemma_as_stem:
    assert not args.lemma_morphemes, 'cannot enable both lemma_as_stem and lemma_morphemes options at once'

if args.debug:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

logger = logging.getLogger('CoNLLPreprocessMorph')


# leave-it-be...because it's on by default and we can't "turn off" this argument itself
if args.posmorph:
    if args.mode != WORD_TO_MORPHTAG:
        logger.info('NOTE: POSMORPH only applicable to word-to-morphtag (mode 1)')

if args.output_breaklevel_flags:
    assert args.mode == UNIT_TO_UNIT or args.mode == MORPH_TO_MORPHTAG, 'breaklevel tags only supported in unit2unit or morph2tag (mode 0 or 2)'
    assert args.output_format == 'seq2seq', 'breaklevel tags only supported in seq2seq output format'

    if args.mode == MORPH_TO_MORPHTAG:
        print('WARNING: breaklevel flags mark only one overlapping morphemic segment as WD-BEGIN')

if args.output_format == 'marmot':
    assert args.mode == WORD_TO_MORPHTAG, 'MarMoT format only applicable to word-to-morphtag (mode 1)'

if args.mode == MORPH_TO_MORPHTAG:
    assert args.lemma_morphemes or args.lemma_as_stem, 'lemma morphemes must be enabled for morph-to-morphtag (mode 2)'

#if args.lemma_morphemes or args.lemma_as_stem:
#    assert args.mode == UNIT_TO_UNIT or args.mode == MORPH_TO_MORPHTAG, 'lemma morphemes only available for char-to-char (mode 0) or per-morpheme tagging mode (mode 2)'

if args.enrich_with_original_word:
    assert args.mode == UNIT_TO_UNIT or args.mode == MORPH_TO_MORPHTAG, \
                                     'original word enrichment only available in unit2unit or per-morpheme tagging mode (mode 0 or 2)'

if args.use_lemma_morphemes_for_morphtag:
    assert args.mode == WORD_TO_MORPHTAG, 'using combined lemma-morphemes for morphtag is only supported for word-to-morphtag (mode 1)'
    assert args.lemma_morphemes, 'using combined lemma-morphemes requires that lemma-morphemes option is enabled'

if args.spmrl:
    assert args.lemma_morphemes, 'SPMRL option only makes sense with lemma ' \
                                 'morphemes option'

assert args.training_split + args.dev_split + args.testing_split == 100

logger.info('Outputting project configuration to configs/%s...' % args.config_name)

try:
    os.mkdir('configs')
except:
    pass

# Warn if config already exists
if os.path.exists('configs/%s' % args.config_name):
    logger.info('Please remove pre-existing config at: %s' % 'configs/%s' %
        args.config_name)
    sys.exit(1)

try:
    os.mkdir('configs/%s' % args.config_name)
except:
    pass

try:
    os.mkdir('configs/%s/data' % args.config_name)
except:
    pass

try:
    os.mkdir('configs/%s/model' % args.config_name)
except:
    pass

sent_error_count = 0
corpus_files = [args.input_file]
all_sentences = []
# HACK
main_corpus = None

class OutputUnit(object):
    def __init__(self):
        # char itself
        self.c = ''
        # can be used as secondary input to DNN to specify word-breaks, etc
        self.flag = ''
        self.action_tag = ''

class OutputMorph(object):
    def __init__(self):
        # used during output
        # morpheme itself
        self.form = ''
        # can be used as secondary input to DNN to specify word-breaks, etc
        self.flag = ''

        # original untransformed word that this morpheme is a part of
        # can be used as auxiliary input to POS tagger to make use of word
        # vectors
        # self.orig_word = ''
        # this class is stored inside a word, so we can use that instead

        self.action_tag = ''

class OutputWord(object):
    def __init__(self):
        # used during output
        # word itself
        self.form = ''
        # can be used as secondary input to DNN to specify word-breaks, etc
        self.flag = ''
        self.action_tag = ''

        # for marmot
        self.pos_tag = ''

        # if in unit2unit2 mode only
        self.output_units = []

        # list of output morphemes in the word
        # mode 2 only
        self.output_morphs = []

    def unitAppend(self, c):
        self.output_units.append(c)

    def morphAppend(self, m):
        self.output_morphs.append(m)

class OutputSentence(object):
    def __init__(self):
        # list of output words in the sentence
        self.output_words = []

        # pointer to orig conll sentence (used during filtering)
        self.orig_conll_sentence = None

    def append(self, w):
        self.output_words.append(w)

# determine if this is punctuation or not
def ispunct(p):
    return unicodedata.category(p).startswith('P')

#spaces
space_tbl = [chr(i) for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('Zs')]

def strip_all_space(s):
    result = ''
    for c in s.strip():
        if c in space_tbl:
            pass
        else:
            result += c
    return result.strip()

def isnum(p):
    return p.isdigit() # TODO: what about strange digits like ^2, ^3, etc(powers of 2,3)
    # isnumeric vs isdigit()

# whole punctuation set is logger.infoed at end so that checking consistency
# is possible

replace_chars = {
    '％': '%',
    '：': ':',
    '；': ';',
    '〔': '(',
    '〕': ')',
    '，': ',',
    '．': '.',
    '－': '―',
    '？': '?',
    '！': '!',
    '　': ' ',   # Unicode Character 'IDEOGRAPHIC SPACE' (U+3000)
    '“': '"',
    '”': '"',
    '‘': '\'',
    '’': '\''
}

def actionToPrettyString(a):
    if type(a) is tuple:
        if a[0] == 'KEEP':
            return 'KEEP'
        elif a[0] == 'NOOP':
            return 'NOOP'
        elif a[0] == 'B-KEEP':
            return 'B-KEEP'
        elif a[0] == 'I-KEEP':
            return 'I-KEEP'
        elif a[0] == 'MOD':
            #return 'MOD:' + ''.join(a[1])
            return 'MOD:' + str(a[1]).replace(', ', ',')
        else:
            assert None, 'unknown action: ' + str(a)
    else:
        assert ' ' not in a
        return a # just trust it

'''
Return the list of actions that are not in the top N for frequencies
'''
def getUncommonActionsToRemove(sentences, N):
    actionsSorted = topNActions(getActionFrequencies(sentences))
    return set([actionName for (actionName, actionFreq) in actionsSorted[N:]])

'''
Filter sentences with actions that are in the blacklist actionsToRemove

Useful for reducing action count with rare forms
'''
def filterSentencesByActionBlacklist(sentences, actionsToRemove):
    assert 0 <= args.mode <= 2 # others not supported yet
    newSentences = []

    if args.mode == UNIT_TO_UNIT:
        for sent in sentences:
            if sent.orig_conll_sentence.evict: # already evicted?
                continue
            evictSentence = False
            for word in sent.output_words:
                for unit in word.output_units:
                    if unit.action_tag in actionsToRemove:
                        evictSentence = True
                        break
                if evictSentence:
                    break
            if not evictSentence:
                newSentences.append(sent)
            else:
                sent.orig_conll_sentence.evict = True
    elif args.mode == WORD_TO_MORPHTAG:
        for sent in sentences:
            if sent.orig_conll_sentence.evict: # already evicted?
                continue
            evictSentence = False
            for word in sent.output_words:
                if word.action_tag in actionsToRemove:
                    evictSentence = True
                    break
            if not evictSentence:
                newSentences.append(sent)
            else:
                sent.orig_conll_sentence.evict = True
    elif args.mode == MORPH_TO_MORPHTAG:
        for sent in sentences:
            if sent.orig_conll_sentence.evict: # already evicted?
                continue
            evictSentence = False
            for word in sent.output_words:
                for morph in word.output_morphs:
                    if morph.action_tag in actionsToRemove:
                        evictSentence = True
                        break
            if not evictSentence:
                newSentences.append(sent)
            else:
                sent.orig_conll_sentence.evict = True

    return newSentences

## TODO:
# after testing these functions, also add ability to remove less frequent
# words and units
#def filterSentencesByWordFreq
#def filterSentencesByUnitFreq

'''
Filter sentences with actions that occur less often than cut-off threshold

Similar to above function, but lets us specify an exact frequency value cut-off
rather than a set of blacklisted actions

Useful for reducing action count with rare forms
'''
def filterSentencesByActionFreq(sentences, threshold):
    assert 0 <= args.mode <= 2 # others not supported yet
    actionFrequencies = getActionFrequencies(sentences)
    newSentences = []

    if args.mode == UNIT_TO_UNIT:
        for sent in sentences:
            if sent.orig_conll_sentence.evict: # already evicted?
                continue
            evictSentence = False
            for word in sent.output_words:
                for unit in word.output_units:
                    if actionFrequencies[unit.action_tag] < threshold:
                        evictSentence = True
                        break
                if evictSentence:
                    break
            if not evictSentence:
                newSentences.append(sent)
            else:
                sent.orig_conll_sentence.evict = True
    elif args.mode == WORD_TO_MORPHTAG:
        for sent in sentences:
            if sent.orig_conll_sentence.evict: # already evicted?
                continue
            evictSentence = False
            for word in sent.output_words:
                if actionFrequencies[word.action_tag] < threshold:
                    evictSentence = True
                    break
            if not evictSentence:
                newSentences.append(sent)
            else:
                sent.orig_conll_sentence.evict = True
    elif args.mode == MORPH_TO_MORPHTAG:
        for sent in sentences:
            if sent.orig_conll_sentence.evict: # already evicted?
                continue
            evictSentence = False
            for word in sent.output_words:
                for morph in word.output_morphs:
                    if actionFrequencies[morph.action_tag] < threshold:
                        evictSentence = True
                        break
                if evictSentence:
                    break
            if not evictSentence:
                newSentences.append(sent)
            else:
                sent.orig_conll_sentence.evict = True

    return newSentences


'''
Get word frequencies for the specified sentence set
'''
def getWordFrequencies(sentences):
    wordFreq = {}

    for sent in sentences:
        for word in sent.output_words:
            if word.form not in wordFreq:
                wordFreq[word.form] = 0
            wordFreq[word.form] += 1

    return wordFreq

'''
Get unit frequencies for the specified sentence set
'''
def getUnitFrequencies(sentences):
    assert 0 <= args.mode <= 1 # others not supported yet
    unitFreq = {}

    if args.mode == UNIT_TO_UNIT:
        for sent in sentences:
            for word in sent.output_words:
                for unit in word.output_units:
                    if unit.c not in unitFreq:
                        unitFreq[unit.c] = 0
                    unitFreq[unit.c] += 1
    elif args.mode == WORD_TO_MORPHTAG:
        for sent in sentences:
            for word in sent.output_words:
                for c in word.form:
                    if c not in unitFreq:
                        unitFreq[c] = 0
                    unitFreq[c] += 1

    return unitFreq

'''
Get action frequencies for the specified sentence set
'''
def getActionFrequencies(sentences):
    assert 0 <= args.mode <= 2 # others not supported yet
    actionFreq = {}

    if args.mode == UNIT_TO_UNIT:
        for sent in sentences:
            for word in sent.output_words:
                for unit in word.output_units:
                    if unit.action_tag not in actionFreq:
                        actionFreq[unit.action_tag] = 0
                    actionFreq[unit.action_tag] += 1
    elif args.mode == WORD_TO_MORPHTAG:
        for sent in sentences:
            for word in sent.output_words:
                if word.action_tag not in actionFreq:
                    actionFreq[word.action_tag] = 0
                actionFreq[word.action_tag] += 1
    elif args.mode == MORPH_TO_MORPHTAG:
        for sent in sentences:
            for word in sent.output_words:
                for morph in word.output_morphs:
                    if morph.action_tag not in actionFreq:
                        actionFreq[morph.action_tag] = 0
                    actionFreq[morph.action_tag] += 1

    return actionFreq

'''
Get the total number of words in the specified sentences
'''
def getIndividualWordCount(sentences):
    wordCount = 0

    for sent in sentences:
        wordCount += len(sent.output_words)

    return wordCount

'''
Get the total number of units in the specified sentences
'''
def getIndividualUnitCount(sentences):
    assert 0 <= args.mode <= 1 # others not supported yet
    unitCount = 0

    for sent in sentences:
        for word in sent.output_words:
            if args.mode == UNIT_TO_UNIT:
                # always use this array instead of iterating wd.FORM
                # just in case we decide not to process units that are still
                # in the FORM
                unitCount += len(word.output_units)
            else:
                # assuming chars for unit in this case
                unitCount += len(word.form)

    return unitCount


'''
Get the total number of morphemes in the specified sentences
'''
def getIndividualMorphCount(sentences):
    assert args.mode == MORPH_TO_MORPHTAG # others not supported yet
    morphCount = 0

    for sent in sentences:
        for word in sent.output_words:
            morphCount += len(word.output_morphs)

    return morphCount


'''
Get the top N actions from the action frequencies dictionary,
in descending order of frequency, or all actions sorted by frequency if N=0

Returns list of
    [(action, freq),
     ...
    ]
'''
def topNActions(actionFreq, N=0):
    assert type(actionFreq) is dict
    assert type(N) == int and N >= 0
    topActionsList = sorted([(v, k) for (k, v) in actionFreq.items()], reverse=True)
    topActionsList_ActionFreqOrder = [(k, v) for (v, k) in topActionsList]

    if N == 0:
        # return all
        return topActionsList_ActionFreqOrder
    else:
        return topActionsList_ActionFreqOrder[:N]

def procFile(fname):
    global main_corpus
    global sent_error_count
    global dupes
    global overlap_counts
    global setidx_to_str

    fd = open(fname, 'r', encoding='utf-8') #, errors='ignore')
    contents = fd.read()
    fd.close()

    retval_sentences = []

    corpus = ConllFile(keepMalformed=True,
                               checkParserConformity=False,
                               projectivize=False,
                               enableLemmaMorphemes=args.lemma_morphemes,
                               enableLemmaAsStem=args.lemma_as_stem,
                               compatibleJamo=True,
                               fixMorphemeLabelBugs=True,
                               spmrlFormat=args.spmrl,
                               koreanAsJamoSeq=args.conll_jamo_split)
    corpus.read(contents)

    main_corpus = corpus

    # pre-scan for errors

    setidx_to_obj = {0: corpus}

    for sentence_idx, sentence in enumerate(corpus):
        sentence.evict = False
        error_found = False

        total_chars = 0

        '''
        ; 21세기 세종계획 균형말뭉치(2000).zip, Written/General/2/CH000093.TXT.tag: sentence 2401
        1       ^A^Bvx^A^A^H    vx/SL   _       _       _       0       _       _       _
        Ignore sentences like this. Something went wrong.
        '''

        '''
        Check for duplicate sentences
        '''

        if not args.keep_duplicates:
            h = hash(sentence)
            if h in dupes:
                dset, didx = dupes[h]
                o = setidx_to_obj[dset]
                if o.sentences[didx].is_equal(sentence):
                    logger.debug('Duplicate data point <%s, all>: %s' % (setidx_to_str[dset], sentence.toSimpleRepresentation()))
                    overlap_counts['<%s>' % ', '.join(sorted(['all', setidx_to_str[dset]]))] += 1

                    sentence.evict = True
                    logger.debug('Skipping sentence')
                    sent_error_count += 1
                    continue
                else:
                    logger.debug('Hash collision', h, sentence.toSimpleRepresentation())
            else:
                dupes[h] = (0, sentence_idx) # training set

        for wd_i, wd in enumerate(sentence.tokens):
            #if 'vx' in wd.FORM:
            #    print('TODO: ', wd.FORM, wd.LEMMA, wd.morphemes)
            #    sys.exit(1)

            if wd.FORM:
                if len(wd.FORM.strip()) == 0 or ' ' in wd.FORM:
                    logger.debug('Error found: type 0')
                    error_found = True

                form_copy = wd.FORM

                # account for underscore count when checking max_chars
                if args.append_underscore_to_word:
                    # if we're not the last word,
                    # add a special whitespace _ char
                    if wd_i < len(sentence.tokens)-1:
                        form_copy += '_'

                for c in form_copy:
                    total_chars += 1
                    if len(strip_all_space(c)) == 0:
                        logger.debug('Error found: type 1')
                        error_found = True
            else:
                error_found = True
                logger.debug('Error found: type 2')

            if not error_found:
                if args.lemma_morphemes or args.lemma_as_stem:
                    if (not wd.LEMMA) or len(wd.LEMMA.strip()) == 0:
                        logger.debug('Error found: type 3.1 ' +
                                     sentence.toFileOutput())
                        error_found = True

                if args.lemma_as_stem:
                    if ' ' in wd.LEMMA:
                        logger.debug('Error found: type 3.2 ' +
                                     sentence.toFileOutput())
                        error_found = True

            if not error_found:
                if args.lemma_morphemes:
                    if args.spmrl:
                        if len(wd.XPOSTAG.strip()) == 0:
                            logger.debug('Error found: type 3.3')
                            error_found = True

                    if not error_found:
                        if wd.morphemes:
                            for m in wd.morphemes:
                                if len(strip_all_space(m[0])) == 0:
                                    logger.debug('Error found: type 3.3.1')
                                    error_found = True
                                if len(strip_all_space(m[1])) == 0:
                                    logger.debug('Error found: type 3.3.2')
                                    error_found = True

        if error_found:
            sentence.evict = True
            logger.debug('Skipping sentence')
            sent_error_count += 1
            continue

        if args.max_char_count != 0:
            # FIXME: would be great if this took into account underscore tokens
            if total_chars > args.max_char_count:
                sentence.evict = True
                logger.debug('Skipping sentence: exceeds max char length')
                sent_error_count += 1
                continue
        
        current_sentence = OutputSentence()
        current_sentence.orig_conll_sentence = sentence

        for wd_i in range(len(sentence.tokens)):
            wd = sentence.tokens[wd_i]
            current_word = OutputWord()
            current_word.form = wd.FORM

            # match same as above code
            if args.append_underscore_to_word:
                # if we're not the last word,
                # add a special whitespace _ char
                if wd_i < len(sentence.tokens)-1:
                    current_word.form += '_'

            if wd.UPOSTAG == None:
                current_word.pos_tag = ''
            else:
                current_word.pos_tag = wd.UPOSTAG.lower()

            if args.mode == UNIT_TO_UNIT:
                # only do expansion

                if args.lemma_morphemes:
                    #segs = [part.rsplit('/', 1)[0] for part in wd.LEMMA.split(' + ')]
                    segs = [m[0] for m in wd.morphemes]
                else:
                    segs = list(wd.LEMMA)

                if args.append_underscore_to_word:
                    # if we're not the last word,
                    # add a special whitespace _ char
                    if wd_i < len(sentence.tokens)-1:
                        segs.append('_')

                if args.u2u_oracle == 0:
                    # smart char2char mode
                    #logger.info('PROCESS', current_word.form, segs)
                    try:
                        a = greedyOracle(current_word.form, segs)
                    except:
                        print('failed', wd_i, current_word.form, segs,
                              sentence.toFileOutput())
                        sys.exit(1)
                    #logger.info(a)
                    action_list = addSegmentation(a, current_word.form, segs)
                    #logger.info(action_list)
                    c = restoreOrigSegments(action_list, current_word.form)
                    assert c == segs, 'restored segments don\'t match: ' + str(c) + ' vs ' + str(segs)
                    assert len(action_list) == len(current_word.form)
                    
                    for i in range(len(current_word.form)):
                        current_unit = OutputUnit()
                        a = action_list[i]
                        f = current_word.form[i]
                        current_unit.c = f

                        #if a[0] == 'B-KEEP' or a[0] == 'I-KEEP':
                        #    a = ('KEEP', ) # crush for now

                        # don't cut off B/I-
                        #if a[0] == 'MOD':
                        #    for item_i in range(len(a[1])):
                        #        a[1][item_i]=a[1][item_i][2:] # cut off B-/I-, only do expansion

                        assert len(strip_all_space(f)) != 0

                        action = actionToPrettyString(a)

                        # special exception for sequence-tagging-morph project
                        #if args.output_format == 'seq2seq':
                        #    action = 'B-'+action

                        current_unit.action_tag = action

                        if args.output_breaklevel_flags:
                            if i == 0:
                                current_unit.flag = 'CHR-WD-BEGIN'
                            else:
                                current_unit.flag = 'CHR-WD-INTER'

                        current_word.unitAppend(current_unit)
                elif args.u2u_oracle == 1:
                    # char n-gram mode
                    # automatically create longest subsequences (prefixes/suffixes) from corpus
                    split_indices = simple_ngram_oracle.get_longest_split_pieces(list(current_word.form), TODO_pieces)
                    #if current_word.form == 'Geleceğin':
                        # debug...
                        #print('split_indices', split_indices)
                        #sys.exit(1)
                        #break
                    FORM_split = simple_ngram_oracle.split_at(list(current_word.form), split_indices)
                    LEMMA_split = simple_ngram_oracle.split_at(list(wd.LEMMA), split_indices)
                    action_list = simple_ngram_oracle.varlen_oracle(FORM_split, LEMMA_split)

                    #print('Without suffixes:', suffixes_stripped)
                    #print('FORM split:', FORM_split)
                    #print('LEMMA split:', LEMMA_split)
                    #print('Real lemma:', wd.LEMMA)
                    #print('FORM->LEMMA oracle actions:', action_list)
                    #print()

                    restored = simple_ngram_oracle.restore_from_actions(FORM_split, action_list)
                    assert wd.LEMMA == restored, 'restored segments don\'t match: ' + str(wd.LEMMA) + ' vs ' + str(restored)

                    # TODO: Needs fixing: if more actions, then need to add some padding
                    if len(action_list) > len(FORM_split):
                        print('@@ TODO: actions exceed FORM_split length (len(LEMMA) > len(FORM))')
                    for i in range(len(FORM_split)):
                        current_unit = OutputUnit()
                        a = action_list[i]
                        f = FORM_split[i]
                        current_unit.c = f

                        assert len(strip_all_space(f)) != 0

                        action = actionToPrettyString(a)

                        # special exception for sequence-tagging-morph project
                        #if args.output_format == 'seq2seq':
                        #    action = 'B-'+action

                        current_unit.action_tag = action

                        if args.output_breaklevel_flags:
                            if i == 0: # if first unit, we are at the beginning of the FORM
                                current_unit.flag = 'CHR-WD-BEGIN'
                            else:
                                current_unit.flag = 'CHR-WD-INTER'

                        current_word.unitAppend(current_unit)
                else:
                    assert None

            elif args.mode == WORD_TO_MORPHTAG:
                if args.use_lemma_morphemes_for_morphtag:
                    #all_morph_tags = [part.rsplit('/', 1)[1].strip() for
                    # part in wd.LEMMA.split(' + ')]
                    all_morph_tags = [m[1] for m in wd.morphemes]
                    action = '-'.join(all_morph_tags).upper()
                else:
                    if args.posmorph:
                        action = '|'.join(['upostag='+current_word.pos_tag]+sorted(wd.FEATS)).lower()
                    else:
                        action = '|'.join(sorted(wd.FEATS)).lower()

                # special exception for sequence-tagging-morph project
                #if args.output_format == 'seq2seq':
                #    action = 'B-'+action

                current_word.action_tag = action

            elif args.mode == MORPH_TO_MORPHTAG:
                #if args.u2u_oracle == 0:
                assert args.lemma_morphemes or args.lemma_as_stem

                if args.lemma_morphemes:
                    #all_parts = [part.rsplit('/', 1) for part in wd.LEMMA.split(' + ')]
                    all_parts = wd.morphemes

                    if args.append_underscore_to_word:
                        # if we're not the last word,
                        # add a special whitespace _ char
                        if wd_i < len(sentence.tokens)-1:
                            all_parts.append(('_', 'SPACE'))
                elif args.lemma_as_stem:
                    assert args.u2u_oracle == 1

                    if args.posmorph:
                        action = '|'.join(['upostag='+current_word.pos_tag]+sorted(wd.FEATS)).lower()
                    else:
                        action = '|'.join(sorted(wd.FEATS)).lower()

                    # special exception for sequence-tagging-morph project
                    #if args.output_format == 'seq2seq':
                    #    action = 'B-'+action

                    split_indices = simple_ngram_oracle.get_longest_split_pieces(list(current_word.form), TODO_pieces)
                    FORM_split = simple_ngram_oracle.split_at(list(current_word.form), split_indices)
                    LEMMA_split = simple_ngram_oracle.split_at(list(wd.LEMMA), split_indices)
                    action_list = simple_ngram_oracle.varlen_oracle(FORM_split, LEMMA_split)

                    restored = simple_ngram_oracle.restore_from_actions_plusnonstem(FORM_split, action_list)

                    #all_parts = [(wd.LEMMA, 'STEM')]
                    #all_parts += [(current_word.form[len(wd.LEMMA):], action)]

                    all_parts = restored
                    assert len(all_parts) > 0

                    for idx, p in enumerate(all_parts):
                        if p[1] == 'NONSTEM':
                            all_parts[idx] = (p[0], action)
                            #p = (p[0], action)
                else:
                    assert None

                for i in range(len(all_parts)):
                    mform, mtag = all_parts[i]
                    current_morph = OutputMorph()
                    current_morph.form = mform

                    #if current_morph.form.isdigit():
                    #    current_morph.form = '$NUM$' # mark specially (done automatically?)

                    action = mtag #.lower()

                    # special exception for sequence-tagging-morph project
                    #if args.output_format == 'seq2seq':
                    #    action = 'B-'+action

                    if args.output_breaklevel_flags:
                        if i == 0: # if first morph, we are at the beginning of the FORM
                            current_morph.flag = 'CHR-WD-BEGIN'
                        else:
                            current_morph.flag = 'CHR-WD-INTER'

                    current_morph.action_tag = action

                    current_word.morphAppend(current_morph)


                '''elif args.u2u_oracle == 1:

                    if args.posmorph:
                        action = '|'.join(['upostag='+current_word.pos_tag]+sorted(wd.FEATS)).lower()
                    else:
                        action = '|'.join(sorted(wd.FEATS)).lower()

                    # take the split morpheme pieces and tag the word's non-stem part with the word's original FEATS
                    # char n-gram mode
                    # automatically create longest subsequences (prefixes/suffixes) from corpus
                    #split_indices = simple_ngram_oracle.get_longest_split_pieces(list(current_word.form), TODO_pieces)
                    #if current_word.form == 'Geleceğin':
                        # debug...
                        #print('split_indices', split_indices)
                        #sys.exit(1)
                        #break
                    #FORM_split = simple_ngram_oracle.split_at(list(current_word.form), split_indices)
                    #LEMMA_split = simple_ngram_oracle.split_at(list(wd.LEMMA), split_indices)
                    #action_list = simple_ngram_oracle.varlen_oracle(FORM_split, LEMMA_split)

                    current_morph = OutputMorph()
                    current_morph.form = wd.LEMMA
                    current_word.morphAppend(current_morph)

                    current_morph = OutputMorph()
                    current_morph.form = current_word.form[len(wd.LEMMA):]
                    current_morph.action_tag = wd.FEATS'''
                    

            if args.min_word_count != 0:
                if len(current_sentence) >= args.min_word_count:
                    current_sentence.append(current_word)
                else:
                    sentence.evict = True
            else:
                current_sentence.append(current_word)

        retval_sentences.append(current_sentence)

    return retval_sentences

for f in corpus_files:
    logger.info('Processing %s...' % f)
    all_sentences += procFile(f)

if args.mode == UNIT_TO_UNIT or args.mode == WORD_TO_MORPHTAG:
    word_count = getIndividualWordCount(all_sentences)
    unit_count = getIndividualUnitCount(all_sentences)
    logger.info('%d sentences found with %d words and %d units' % (len(all_sentences), word_count, unit_count))
elif args.mode == MORPH_TO_MORPHTAG:
    word_count = getIndividualWordCount(all_sentences)
    morph_count = getIndividualMorphCount(all_sentences)
    logger.info('%d sentences found with %d words and %d morphemes' % (len(all_sentences), word_count, morph_count))

overlap_count_sum = sum(overlap_counts.values())

logger.info('%d duplicate sentences ignored (already subtracted from above total)' % (overlap_count_sum))

if not args.keep_duplicates:
    logger.info('... overlap statistics: %s' % (overlap_counts))

logger.info('%d erroneous sentences ignored (including duplicates, already subtracted from above total)' % (sent_error_count))

action_frequencies = getActionFrequencies(all_sentences)
logger.info('%d unique action tags found across all sets' % len(action_frequencies.keys()))

sorted_frequencies = topNActions(action_frequencies)
logger.debug('All tag frequencies:')
for (k, v) in sorted_frequencies:
    logger.debug('%s\t%d' % (k, v))

if args.shuffle:
    logger.info('Shuffling all sentences...')
    random.shuffle(all_sentences)

train_sentences = []
test_sentences = []
dev_sentences = []

cross_validation = False

for sent_idx, sent in enumerate(all_sentences):
    # preserve orig index for later
    sent.internal_idx = sent_idx

if args.training_split == 100:
    # disable cross-validation and output all
    train_sentences = all_sentences
else:
    cross_validation = True

    train_fraction = float(args.training_split) / 100.0
    dev_fraction = float(args.dev_split) / 100.0
    test_fraction = float(args.testing_split) / 100.0

    training_count = int(len(all_sentences)*train_fraction)
    testing_count = int(len(all_sentences)*test_fraction)

    # must at least have training and testing sets for cross-validation
    assert training_count > 0 and testing_count > 0

    train_sentences = all_sentences[:training_count]
    all_sentences = all_sentences[training_count:]

    if dev_fraction > 0:
        # even if testing_count exceeds length of array, no problem when slicing

        test_sentences = all_sentences[:testing_count]
        all_sentences = all_sentences[testing_count:]

        # remaining ones
        dev_sentences = all_sentences[:]

        assert len(test_sentences) > 0
        assert len(dev_sentences) > 0
    else:
        # remaining ones
        test_sentences = all_sentences[:]

        assert len(test_sentences) > 0

'''
Get the proper file output string for all of the specified sentences
'''
def getFileOutput(sentences):
    assert 0 <= args.mode <= 2 # others not supported yet
    assert args.output_format in ['seq2seq', 'marmot'] # others not supported yet
    fileOutput = ''

    if args.output_format == 'seq2seq':
        outSentences = []
        if args.mode == UNIT_TO_UNIT:
            for sent in sentences:
                if sent.orig_conll_sentence.evict:
                    continue
                curSentence = []
                for word in sent.output_words:
                    for unit in word.output_units:
                        origword = ''
                        if args.enrich_with_original_word:
                            origword = word.form
                        flag = ''
                        if args.output_breaklevel_flags:
                            flag = unit.flag

                        curSentence.append('%s\t%s\t%s\t%s' % (unit.c,
                                                             origword,
                                                             flag,
                                                             unit.action_tag))

                outSentences.append(curSentence)
        elif args.mode == WORD_TO_MORPHTAG:
            for sent in sentences:
                if sent.orig_conll_sentence.evict:
                    continue
                curSentence = []
                for word in sent.output_words:
                    curSentence.append('%s\t\t\t%s' % (word.form,
                                                       word.action_tag))
                outSentences.append(curSentence)
        elif args.mode == MORPH_TO_MORPHTAG:
            for sent in sentences:
                if sent.orig_conll_sentence.evict:
                    continue
                curSentence = []
                for word in sent.output_words:
                    for morph in word.output_morphs:
                        origword = ''
                        if args.enrich_with_original_word:
                            origword = word.form
                        flag = ''
                        if args.output_breaklevel_flags:
                            flag = morph.flag

                        curSentence.append('%s\t%s\t%s\t%s' % (morph.form,
                                                               origword,
                                                               flag,
                                                               morph.action_tag))

                outSentences.append(curSentence)

        fileOutput = '\n\n'.join(['\n'.join(sent) for sent in outSentences])
    elif args.output_format == 'marmot':
        outSentences = []

        # MarMoT isn't really used for char2char mode

        assert args.mode == WORD_TO_MORPHTAG, 'other output for MarMoT not implemented'

        '''
        if args.mode == UNIT_TO_UNIT:
            for sent in sentences:
                curSentence = []
                for word in sent.output_words:
                    for char_i in range(len(word.output_units)):
                        char = word.output_units[char_i]
                        # for now , CHAR can be used as the POS tag??? unsure of purpose here
                        curSentence.append('%d\t%s\t%s\t%s' % (char_i+1, char.c, 'CHAR', char.action_tag))

                outSentences.append(curSentence)
        elif args.mode == WORD_TO_MORPHTAG:
        '''

        if args.mode == WORD_TO_MORPHTAG:
            for sent in sentences:
                if sent.orig_conll_sentence.evict:
                    continue
                curSentence = []
                for word_i in range(len(sent.output_words)):
                    word = sent.output_words[word_i]
                    pos_tag = word.pos_tag
                    if not pos_tag:
                        pos_tag = 'none'
                    curSentence.append('%d\t%s\t%s\t%s' % (word_i+1, word.form, word.pos_tag, word.action_tag))
                outSentences.append(curSentence)

        fileOutput = '\n\n'.join(['\n'.join(sent) for sent in outSentences])

    return fileOutput


if args.max_actions_training != 0:
    if cross_validation:
        logger.info('Performing action trimming on the training set')
    else:
        logger.info('Performing action trimming')

    ev_train = 0
    for sent in train_sentences:
        if sent.orig_conll_sentence.evict:
            ev_train += 1

    old_count = len(train_sentences)-ev_train
    train_action_count = len(getActionFrequencies(train_sentences).keys())
    action_blacklist = getUncommonActionsToRemove(train_sentences, args.max_actions_training)
    logger.info('Blacklisting %d/%d uncommon training actions' % (len(action_blacklist), train_action_count))
    included_train_sentences = filterSentencesByActionBlacklist(train_sentences, action_blacklist)
    # included after evicting all existing sentences marked for eviction
    
    new_count = len(included_train_sentences)
    # only used for counting: later, detect sentence.evict property for filtering

    ## HACK: also re-output sentences filtered
    #global main_corpus
    #with open('/tmp/filtered.conllu', 'w', encoding='utf-8') as fd:
    #    for sent in main_corpus.sentences:
    #        if sent.evict == False:
    #            fd.write(sent.toFileOutput() + '\n' + '\n')

    if cross_validation:
        logger.info('Non-errored training set: %d => %d sentences using top %d actions' % (old_count, new_count, args.max_actions_training))
    else:
        logger.info('Non-errored set: %d => %d sentences using top %d actions' % (old_count, new_count, args.max_actions_training))


if args.ignore_training_corpus_sent_ids != None:
    orig_id_to_train_sent_id = dict()

    # map id of original corpus entry to id in training set
    for train_sent_idx, train_sent in enumerate(train_sentences):
        orig_id_to_train_sent_id[train_sent.internal_idx] = train_sent_idx

    filter_sent_orig_ids = []
    with open(args.ignore_training_corpus_sent_ids, 'r', encoding='utf-8') as fd:
        tmp = fd.read().split('\n')
        for ln in tmp:
            if not ln:
                continue
            filter_sent_orig_ids.append(int(ln))

    for sent_id in filter_sent_orig_ids:
        train_sentences[orig_id_to_train_sent_id[sent_id]].orig_conll_sentence.evict = True

    logger.info('Training set: evicted %d sentences from filter input file' % (len(filter_sent_orig_ids)))

# NOTE: don't shuffle when using this function
if args.output_evicted_training_sent_ids != None:
    ev_count = 0
    with open(args.output_evicted_training_sent_ids, 'w', encoding='utf-8') as fd:
        for sent in train_sentences:
            if sent.orig_conll_sentence.evict:
                # print original sentence ID before 
                fd.write('%d\n' % sent.internal_idx)
                ev_count += 1

    logger.info('Training set: output %d evicted sentences to filter output file' % (ev_count))

ev_train = 0
for sent in train_sentences:
    if sent.orig_conll_sentence.evict:
        ev_train += 1

if cross_validation:
    ev_test = 0
    ev_dev = 0

    for sent in test_sentences:
        if sent.orig_conll_sentence.evict:
            ev_test += 1

    for sent in dev_sentences:
        if sent.orig_conll_sentence.evict:
            ev_dev += 1

    logger.info('Pre-eviction cross-validation: %d/%d/%d split' % (args.training_split, args.testing_split, args.dev_split))
    logger.info('Pre-eviction: %d training sentences, %d original testing sentences, %d original dev sentences' % (len(train_sentences), len(test_sentences), len(dev_sentences)))
    logger.info('Final: %d training sentences, %d testing sentences, %d dev sentences' % (len(train_sentences)-ev_train, len(test_sentences)-ev_test, len(dev_sentences)-ev_dev))

    all_sent_count = len(train_sentences)+len(test_sentences)+len(dev_sentences)-ev_train-ev_test-ev_dev

    final_train_split = 100.0*float(len(train_sentences)-ev_train) / float(all_sent_count)
    final_test_split = 100.0*float(len(test_sentences)-ev_test) / float(all_sent_count)
    final_dev_split = 100.0*float(len(dev_sentences)-ev_dev) / float(all_sent_count)

    logger.info('Final split: %.2f/%.2f/%.2f' % (final_train_split, final_test_split, final_dev_split))
    #if args.max_actions_training != 0:
        #logger.info('... NOTE: training sentence count is post-action-trim')
        #logger.info('... NOTE: sentence counts are pre-eviction')
else:
    logger.info('Pre-eviction: %d sentences' % (len(train_sentences)))
    logger.info('Final: %d sentences' % (len(train_sentences)-ev_train))
    #if args.max_actions_training != 0:
    #    logger.info('... NOTE: sentence count is post-action-trim')


config_train_filename = ''
config_dev_filename = ''
config_testing_filename = ''

if cross_validation:
    #avail_sets = [('train', train_sentences)]
    #if len(test_sentences) > 0:
    #    avail_sets.append(('test', test_sentences))
    #if len(dev_sentences) > 0:
    #    avail_sets.append(('dev', dev_sentences))

    # just output all no matter how many sentences are in each...
    # we don't want to be left with an old dev-set of an old run if this run's
    # dev set size is 0, for example

    #for (k, v) in avail_sets:
    for (k, v) in [('train', train_sentences), ('test', test_sentences), ('dev', dev_sentences)]:
        output_fn = 'configs/%s/data/%s-%s-%s.txt' % (args.config_name, mode_to_str[args.mode], args.output_format, k)
        if k == 'train':
            config_train_filename = output_fn
        elif k == 'dev':
            config_dev_filename = output_fn
        elif k == 'test':
            config_test_filename = output_fn
        logger.info('Write: %s' % output_fn)
        f = open(output_fn, 'w', encoding='utf-8')
        f.write(getFileOutput(v))
        f.close()

    if args.write_config:
        if len(config_train_filename) == 0:
            logger.info('Config parameter may require manual adjustment: no training file')

        if len(config_dev_filename) == 0:
            logger.info('Config parameter may require manual adjustment: no dev file')

        if len(config_test_filename) == 0:
            logger.info('Config parameter may require manual adjustment: no test file')
else:
    #output_fn = '%s-%s-%s-%s.txt' % (mode_to_str[args.mode], 'all', args.output_suffix, args.output_format)
    output_fn = 'configs/%s/data/%s-%s-%s.txt' % (args.config_name, mode_to_str[args.mode], args.output_format, 'all')
    # just set all the same for now; may require manual adjustment

    if args.write_config:
        config_train_filename = output_fn
        config_dev_filename = output_fn
        config_test_filename = output_fn
        logger.info('Config parameter may require manual adjustment: train, dev, test set to all the same file')

    logger.info('Write: %s' % output_fn)
    f = open(output_fn, 'w', encoding='utf-8')
    f.write(getFileOutput(train_sentences))
    f.close()

# anyways, running build_data.py seems necessary
# runs build_data
#from build_data import *
'''
if args.output_format == 'seq2seq':
    # also output ONLY training vocabulary (because we need to leave some OOV in the dev and test sets)
    logger.info('Outputting training lexicon...')

    # TODO: for char2char, we might consider words to be chars...
    #if args.mode == UNIT_TO_UNIT:...

    # sort to keep order consistent
    # TODO: check that equal frequencies remain in same order. may need some work.
    # but if alphabetical sorting is equal each time, should be fine...

    # special exceptions for sequence-tagging-morph project
    #word_list = sorted(set(['$UNK$', '$NUM$']+list(getWordFrequencies(train_sentences).keys())))
    word_list = sorted(getWordFrequencies(train_sentences).keys())
    char_list = sorted(getCharFrequencies(train_sentences).keys())
    # special exception for sequence-tagging-morph project
    #action_list = sorted(set(['O']+list(getActionFrequencies(train_sentences).keys())))
    action_list = sorted(getActionFrequencies(train_sentences).keys())

    # anyways, running build_data.py seems necessary
    
    for (k, v) in [('words', word_list), ('chars', char_list), ('tags', action_list)]:
        output_fn = '%s-%s-%s.txt' % (mode_to_str[args.mode], k, args.output_suffix)
        logger.info('Write: %s' % output_fn)
        f = open(output_fn, 'w', encoding='utf-8')
        f.write('\n'.join(v))
        f.close()
'''

if args.write_config:
    config_template = '''class config():
    # word embedding dimensions are inferred from specified file
    # INPUTS
    # NOTE: for fasttext_filename we assume the first line is a HEADER
    # and other lines are (example for 300-dim word embeddings):
    # word float1 float2 ... float300
    fasttext_filename = "{FASTTEXT_FILENAME}"
    train_filename = "{TRAIN_FILENAME}"
    dev_filename = "{DEV_FILENAME}"
    test_filename = "{TEST_FILENAME}"

    # OUTPUTS
    trimmed_filename = "{TRIMMED_VEC_FILENAME}"
    words_filename = "{WORDS_FILENAME}"
    tags_filename = "{TAGS_FILENAME}"
    chars_filename = "{CHARS_FILENAME}"
    output_path = "{MODEL_PATH}"
    model_output = output_path + "weights/"
    log_path = output_path + "log.txt"

    max_iter = None

    # NOTE: should re-run build_data.py even if changing this value!
    lowercase = False

    # COLUMN 1: PRIMARY UNIT OR WORDS
    # train embeddings of data column 1 (usually word)
    # whether to load in the pretrained fastText embeddings
    # if we load these, they must be of the same dimension (embedding size)
    col1_load_pretrained_embeddings = {COL1_PRETRAINED_EMBEDDINGS}
    # if not loaded, initialized as random_normal

    # whether to train embeddings (even if pre-trained, they get tuned)
    # or whether to freeze the whole embedding layer
    col1_train_embeddings = True
    # size for embedding entirety of col1 (usually a word, but could be a
    # char or morpheme)
    col1_embedding_size = {COL1_EMBEDDING_SIZE}

    # only enable char embedding if col1 is a WORD or MORPHEME
    # if col1 is a CHAR, it doesn't make sense, because embedding
    # is already taking place on col1
    # char embedding places each char in col1 through an LSTM

    col1_char_embedding = {COL1_CHAR_EMBEDDING} # if char embedding, training is 3.5x slower
    # size for embedding of each char within col1
    col1_char_embedding_size = 100
    # hidden size of each direction of the Bi-LSTM that processes the chars in
    # col1
    col1_char_hidden_size = 100

    # COLUMN 2: ORIGINAL WORDS
    # the pretrained embeddings will be word embeddings
    # only useful if col2 is used as 'original word'
    col2_load_pretrained_embeddings = {COL2_PRETRAINED_EMBEDDINGS}
    # train embeddings of data column 2 (usually original word or simply
    # break-level/argument)
    col2_train_embeddings = {COL2_TRAIN_EMBEDDINGS}
    # if col2 is only an argument, making embedding size 1 is probably
    # sufficient
    col2_embedding_size = {COL2_EMBEDDING_SIZE}
    
    # COLUMN 3: ATTRIBUTES {CHR-WD-BEGIN, CHR-WD-INTER}
    col3_train_embeddings = {COL3_TRAIN_EMBEDDINGS}
    col3_embedding_size = {COL3_EMBEDDING_SIZE}

    # hidden size of each direction of the LAST Bi-LSTM before the CRF layer
    # which ultimately combines col1, col2 and col1 char embeddings
    combined_hidden_size = 300

    nepochs = 10
    dropout = 0.7 # 0.7 usually works
    batch_size = 16
    lr = 0.001
    lr_decay = 0.9
    nepoch_no_imprv = 3

    crf = True # if crf, training is 1.7x slower'''

    model_rel_path = 'configs/%s/model/' % args.config_name

    config_template = config_template.replace('{TRAIN_FILENAME}', \
        config_train_filename)
    config_template = config_template.replace('{DEV_FILENAME}', \
        config_dev_filename)
    config_template = config_template.replace('{TEST_FILENAME}', \
        config_test_filename)

    words_lexicon_filename = 'configs/%s/data/words.lexicon' % args.config_name
    tags_lexicon_filename = 'configs/%s/data/tags.lexicon' % args.config_name
    chars_lexicon_filename = 'configs/%s/data/chars.lexicon' % args.config_name

    # Remove existing lexicon files if they exist
    # (Now unnecessary due to existing config warning at beginning)
    '''
    try:
        os.remove(words_lexicon_filename)
    except:
        pass
    try:
        os.remove(tags_lexicon_filename)
    except:
        pass
    try:
        os.remove(chars_lexicon_filename)
    except:
        pass
    '''

    config_template = config_template.replace('{WORDS_FILENAME}', \
        words_lexicon_filename)
    config_template = config_template.replace('{TAGS_FILENAME}', \
        tags_lexicon_filename)
    config_template = config_template.replace('{CHARS_FILENAME}', \
        chars_lexicon_filename)

    config_template = config_template.replace('{MODEL_PATH}', \
        model_rel_path)

    config_template = config_template.replace('{FASTTEXT_FILENAME}', \
        args.config_fasttext)

    preload_embeddings = False
    if len(args.config_fasttext) > 0:
        preload_embeddings = True
        config_template = config_template.replace('{TRIMMED_VEC_FILENAME}', \
            'configs/%s/data/vec.trimmed.npz' % args.config_name)
    else:
        config_template = config_template.replace('{TRIMMED_VEC_FILENAME}', \
            '')

    '''
    UNIT_TO_UNIT = 0
    WORD_TO_MORPHTAG = 1
    MORPH_TO_MORPHTAG = 2
    '''

    # is col1 always just one char?
    col1_is_char = False
    if args.mode == UNIT_TO_UNIT:
        if args.u2u_oracle == 0:
            # makes no sense to do char embeddings
            col1_is_char = True

    if col1_is_char:
        # makes no sense to load pre-trained word embeddings for a char column
        config_template = config_template.replace('{COL1_PRETRAINED_EMBEDDINGS}', \
            'False')
        # don't do char embedding because we already have just one char in col1
        # (in other words, it would be redundant if we did)
        config_template = config_template.replace('{COL1_CHAR_EMBEDDING}', \
            'False')
    else:
        config_template = config_template.replace('{COL1_PRETRAINED_EMBEDDINGS}', \
            str(preload_embeddings))
        # do char embedding because we have a word or morph or n-gram in col1
        config_template = config_template.replace('{COL1_CHAR_EMBEDDING}', \
            'True')

    config_template = config_template.replace('{COL1_EMBEDDING_SIZE}', \
        str(args.config_col1_embed_size))

    if args.enrich_with_original_word:
        config_template = config_template.replace('{COL2_TRAIN_EMBEDDINGS}', \
            'True')
        # we may want to use pre-trained embeddings here
        config_template = config_template.replace('{COL2_PRETRAINED_EMBEDDINGS}', \
            str(preload_embeddings))
        config_template = config_template.replace('{COL2_EMBEDDING_SIZE}', \
            str(args.config_col2_embed_size))
    else:
        # there is no column 2
        config_template = config_template.replace('{COL2_TRAIN_EMBEDDINGS}', \
            'False')
        config_template = config_template.replace('{COL2_PRETRAINED_EMBEDDINGS}', \
            'False')
        config_template = config_template.replace('{COL2_EMBEDDING_SIZE}', \
            '1')

    # column 3 usually just contains two attributes. embedding size 1 is
    # sufficient for this
    config_template = config_template.replace('{COL3_EMBEDDING_SIZE}', \
                                              '1')

    if args.output_breaklevel_flags:
        config_template = config_template.replace('{COL3_TRAIN_EMBEDDINGS}', \
                                                  'True')
    else:
        # there is no column 3
        config_template = config_template.replace('{COL3_TRAIN_EMBEDDINGS}', \
                                                  'False')

    output_fn = 'configs/%s/config.py' % args.config_name
    logger.info('Write: %s' % output_fn)
    f = open(output_fn, 'w', encoding='utf-8')
    f.write(config_template)
    f.close()

    logger.info('To train, run: python3 build_data.py %s' % output_fn)
    logger.info('             : python3 main.py %s' % output_fn)

    logger.info('For tensorboard, run: tensorboard --logdir %s' % os.path.abspath(model_rel_path))
