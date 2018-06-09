# Grade an evaluated model based on its data set
# Make sure to run actions on morph output first to generate tag input
# Workflow:
# python3 model_new.py morph small_500actions_400len_ver2 --batch-size=64
# python3 model_new.py morph small_500actions_400len_ver2 --batch-size=64 --evaluate
# python3 model_new.py tag small_500actions_400len_ver2 --batch-size=64
# python3 run_actions.py small_500actions_400len_ver2_morph_bilstm_cudnn_dynamic_ES300_CH300_DK70_SEQ400_B64_EP100/output/eval_test.out small_500actions_400len_ver2_morph_bilstm_cudnn_dynamic_ES300_CH300_DK70_SEQ400_B64_EP100/output/eval_test.taginput
# python3 model_new.py tag small_500actions_400len_ver2 --batch-size=64 --evaluate --eval-inference-file small_500actions_400len_ver2_morph_bilstm_cudnn_dynamic_ES300_CH300_DK70_SEQ400_B64_EP100/output/eval_test.taginput
# python3 model_grade.py small_500actions_400len_ver2 small_500actions_400len_ver2_morph_bilstm_cudnn_dynamic_ES300_CH300_DK70_SEQ400_B64_EP100/output/eval_test.out   small_500actions_400len_ver2_tag_bilstm_cudnn_dynamic_ES300_CH300_DK70_SEQ400_B64_EP100/output/inference.out


import sys
import argparse
from bio_tools import *
from logging import handlers

parser = argparse.ArgumentParser(description='Grade two-stage morphing/tagging model in an end-to-end fashion.')

# forced to be morph or tag for now
#parser.add_argument('model_type', type=str,
#                    help='morph or tag')
parser.add_argument('model_data_dir', type=str,
                    help='Data directory containing test data (data/model_data_dir/model_type_{train,dev,test}.txt')
parser.add_argument('morph_eval', type=str, default=None,
                    help='Path to load morph eval file from (inference of testing set)')
parser.add_argument('tag_eval', type=str, default=None,
                    help='Path to load tag eval file from (inference of morph stage output)')
args = parser.parse_args()

if args.model_data_dir.endswith('/'):
    args.model_data_dir = args.model_data_dir[:-1]

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : [%(name)s] : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('Grade')

logger.info('Morph gold input: data/%s/%s_test.txt' % (args.model_data_dir, 'morph'))
logger.info('Morph inference input: %s' % args.morph_eval)
logger.info('Tag gold input: data/%s/%s_test.txt' % (args.model_data_dir, 'tag'))
logger.info('Tag inference input: %s' % args.tag_eval)

morph_gold = BIODataInput('data/%s/%s_test.txt' % (args.model_data_dir, 'morph'))
morph_eval = BIODataInput(args.morph_eval)

tag_gold = BIODataInput('data/%s/%s_test.txt' % (args.model_data_dir, 'tag'))
tag_eval = BIODataInput(args.tag_eval)

# length checking
assert len(morph_gold.sentences) == len(morph_eval.sentences)
assert len(tag_gold.sentences) == len(tag_eval.sentences)
assert len(morph_eval.sentences) == len(tag_eval.sentences)

sentence_count = len(morph_gold.sentences)

logger.info('Length check passed (%d sentences detected)' % (sentence_count))
logger.info('Finding word-level OOV tokens by comparing training set and testing set lexicon...')

'''
# proof of word-level OOV finder... "휩싸였다."
andy@andy-home:/mnt/deeplearn/newunitag$ grep -C3 휩 data/small_500actions_400len_ver2/morph_train.txt|grep -C3 싸
andy@andy-home:/mnt/deeplearn/newunitag$ grep -C3 휩 data/small_500actions_400len_ver2/morph_test.txt|grep -C3 싸
에			B-KEEP
_			B-KEEP
휩			B-KEEP
싸			I-KEEP
였			MOD:['I-이','B-었']
다			B-KEEP


# proof of word-level OOV finder... "쇄신하고자"
andy@andy-home:/mnt/deeplearn/newunitag$ grep -C5 쇄 data/small_500actions_400len_ver2/morph_train.txt|grep -C5 신|grep -C5 고|grep -C5 자
andy@andy-home:/mnt/deeplearn/newunitag$ grep -C5 쇄 data/small_500actions_400len_ver2/morph_test.txt|grep -C5 신|grep -C5 고|grep -C5 자
_			B-KEEP
쇄			B-KEEP
신			I-KEEP
하			B-KEEP
고			B-KEEP
자			I-KEEP
_			B-KEEP
'''



'''
force_segmod_keep: force all space-separated segments to be B,I,I,I... and KEEP actions
                   this includes force_mod_keep automatically
force_mod_keep:    force all MODs or NOOPs to become KEEP actions
'''
def as_word_level(chunks, force_segmod_keep=False, force_mod_keep=False):
    if force_segmod_keep:
        assert not force_mod_keep, 'mutually exclusive options: force_segmod_keep already does force_mod_keep'

    words = []

    current_word = []
    for c_idx, c in enumerate(sorted(chunks)): # make sure to traverse in order
        if c[1].rsplit('/', 1)[0] == '_':   # '_' is reserved
            if len(current_word) > 0:
                words.append(current_word)
                current_word = []
        else:
            if force_segmod_keep:
                morph, tag = c[1].rsplit('/', 1)

                if len(current_word) == 0:
                    current_word.append((len(current_word), '/'.join((morph, 'B-KEEP'))))
                else:
                    current_word.append((len(current_word), '/'.join((morph, 'I-KEEP'))))
            elif force_mod_keep:
                morph, tag = c[1].rsplit('/', 1)
                if tag == 'NOOP':
                    if len(current_word) == 0:
                        # if it was a NOOP action, make it a KEEP depending
                        # on where we are in the word
                        tag = 'B-KEEP'
                    else:
                        tag = 'I-KEEP'
                elif tag == 'B-KEEP' or tag == 'I-KEEP':
                    # no change
                    pass
                else:
                    # mod action that could contain multiple B/I
                    # in our case, just force depending on word position
                    if len(current_word) == 0:
                        # if it was a NOOP action, make it a KEEP depending
                        # on where we are in the word
                        tag = 'B-KEEP'
                    else:
                        tag = 'I-KEEP'

                current_word.append((len(current_word), '/'.join((morph, tag))))
            else:
                current_word.append((len(current_word), c[1]))

    if len(current_word) > 0:
        words.append(current_word)

    return words

'''
Get concatenated morph from word chunk
(results in original input word)

[(0, '(/B-KEEP'), (1, '6/B-KEEP'), (2, ')/B-KEEP')]
to
'(6)'
'''
def morph_only_fromword(l):
    m_out = ''
    for elem in sorted(l):
        morph, tag = elem[1].rsplit('/', 1)
        m_out += morph
    return m_out


# wordlevel_oov_tokens
# (words encountered at the morph stage in the test set that are not 
# present in the training set)
# requires opening up the gold training corpus
wordlevel_oov_token_set = set()
morph_gold_train = BIODataInput('data/%s/%s_train.txt' % (args.model_data_dir, 'morph'))

wordlevel_train_token_set = set()
wordlevel_test_token_set = set()

# determine train set word lexicon
for i in range(len(morph_gold_train.sentences)):
    m_gold_train = morph_gold_train.sentences[i]
    m_gold_train_chunks = m_gold_train.get_input_and_label_chunks()
    m_gold_train_chunks_wordlevel = as_word_level(m_gold_train_chunks)
    for w_chunk in sorted(m_gold_train_chunks_wordlevel):
        wordlevel_train_token_set.add(morph_only_fromword(w_chunk))

# determine test set word lexicon
for i in range(len(morph_gold.sentences)):
    m_gold = morph_gold.sentences[i]
    m_gold_chunks = m_gold.get_input_and_label_chunks()
    m_gold_chunks_wordlevel = as_word_level(m_gold_chunks)
    for w_chunk in sorted(m_gold_chunks_wordlevel):
        wordlevel_test_token_set.add(morph_only_fromword(w_chunk))

wordlevel_oov_token_set = (wordlevel_train_token_set|wordlevel_test_token_set) - wordlevel_train_token_set

logger.info('%d OOV word-level tokens identified' % len(wordlevel_oov_token_set))

# correct sentence-level morph chunks (sentence-level includes proper detection of space tokens)
all_correct_m_chunks = 0
all_total_m_chunks = 0

# correct sentence-level end-to-end chunks (sentence-level includes proper detection of space tokens)
all_correct_e2e_chunks = 0
all_total_e2e_chunks = 0

# correct word-level morph chunks
all_correct_m_chunks_wordlevel = 0
all_total_m_chunks_wordlevel = 0

# correct oov word-level morph chunks
all_correct_m_chunks_wordlevel_oov = 0
all_total_m_chunks_wordlevel_oov = 0

# correct word-level morph chunks (null segment and mod actions)
all_correct_m_chunks_wordlevel_null_segmod = 0
all_total_m_chunks_wordlevel_null_segmod = 0

# correct word-level morph chunks (null mod actions)
all_correct_m_chunks_wordlevel_null_mod = 0
all_total_m_chunks_wordlevel_null_mod = 0

# correct word-level end-to-end chunks (Eojeol level end-to-end accuracy)
all_correct_e2e_chunks_wordlevel = 0
all_total_e2e_chunks_wordlevel = 0

# correct word-level end-to-end chunks among OOV words
all_correct_e2e_chunks_wordlevel_oov = 0
all_total_e2e_chunks_wordlevel_oov = 0

# correct words (probably best metric: atomic, incremented if all morphemes in an entire word is correct end-to-end)
all_correct_words = 0
all_total_words = 0

# correct OOV words
all_correct_words_oov = 0
all_total_words_oov = 0

# correct sentences (atomic, incremented if all words in a sentence are correct end-to-end)
all_correct_sentences = 0
all_total_sentences = 0


#def convert_to_space_level_chunk_array(m_g_chunks, m_e_chunks, t_g_chunks, t_e_chunks):
    
'''
Given input, generate all null keep actions

Make all actions separated by _ to be B,I..
'''
'''def to_null_keep_action_chunks(gold_chunks):
    null_chunks = []

    ## gold_chunks : {(64, "였/MOD:['B-이','B-었']"), (7, '_/B-KEEP'),}...
    ## null_chunks : {(64, "였/MOD:B-KEEP"), (7, '_/B-KEEP'),}...
    for elem in sorted(gold_chunks):
        if

    print(gold_chunks)

    return null_chunks'''

for i in range(sentence_count):
    m_gold = morph_gold.sentences[i]
    m_gold_chunks = m_gold.get_input_and_label_chunks()
    m_eval = morph_eval.sentences[i]
    m_eval_chunks = m_eval.get_input_and_label_chunks()

    #m_eval_chunks_keep = to_null_keep_action_chunks(m_gold_chunks)
    #break

    assert len(m_gold.labels) == len(m_eval.labels)

    t_gold = tag_gold.sentences[i]
    t_gold_chunks = t_gold.get_input_and_label_chunks()
    t_eval = tag_eval.sentences[i]
    t_eval_chunks = t_eval.get_input_and_label_chunks()

    # tags might be different length depending on morphing actions taken!
    # that's what makes grading more difficult
    # assert len(t_gold.labels) == len(t_eval.labels)
    # check the gold tag label chunks against the eval tag label chunks
    # (index, tagged_output)
    # get end-to-end-chunks
    #t_eval_chunks
    #print(t_eval_chunks)

    # it's better to split them up by word first.
    # otherwise we're doing sentence-level chunk accuracy
    # and one wrong word propagates

    m_gold_chunks_wordlevel = as_word_level(m_gold_chunks)
    m_eval_chunks_wordlevel = as_word_level(m_eval_chunks)

    # assume that segmentation and transformation actions are null
    m_eval_chunks_wordlevel_null_segmod = as_word_level(m_eval_chunks, force_segmod_keep=True)
    # assume that segmentation actions are as good as original predicts but that
    # transformation actions are null (all KEEP)
    m_eval_chunks_wordlevel_null_mod = as_word_level(m_eval_chunks, force_mod_keep=True)

    t_gold_chunks_wordlevel = as_word_level(t_gold_chunks)
    t_eval_chunks_wordlevel = as_word_level(t_eval_chunks)

    #print('m_gold_chunks_wordlevel', sorted(m_gold_chunks_wordlevel))
    #print('m_eval_chunks_wordlevel_null_segmod', sorted(m_eval_chunks_wordlevel_null_segmod))
    #break

    ## TODO: make sure this doesn't error for any other reason
    try:
        # just an character-level action-based tag, so should be equal length at
        # morphing stage
        assert len(m_gold_chunks_wordlevel) == len(m_eval_chunks_wordlevel), 'sentence ' + str(i) + ' : ' + str(m_gold_chunks_wordlevel) + '   VS   ' +  str(m_eval_chunks_wordlevel)
        # at tagging stage, unit count may differ and spaces may get deleted somehow
    except:
        # this can fail if special token _ is used. just remove off our grade for now; it's such an edge case.
        '''       
        AssertionError: sentence 39982 :
        [[(0, '왜/B-KEEP')], [(0, '그/B-KEEP'), (1, "_/MOD:['B-/','I-S','I-S','I-X','B-_']"), (2, '모/B-KEEP'), (3, '르/I-KEEP'), (4, '잖/B-KEEP'), (5, '아/I-KEEP'), (6, '요/I-KEEP'), (7, './B-KEEP')]]
        [[(0, '왜/B-KEEP')], [(0, '그/B-KEEP')], [(0, '모/B-KEEP'), (1, '르/I-KEEP'), (2, '잖/B-KEEP'), (3, '아/I-KEEP'), (4, '요/I-KEEP'), (5, './B-KEEP')]]
        '''
        pass



    # evaluate based on NULL segmod morph word-level chunk accuracy (look at each word)
    for w_idx in range(len(m_gold_chunks_wordlevel)):
        # compare inner word chunks
        gold_chunks_inside_word = set(m_gold_chunks_wordlevel[w_idx])
        eval_chunks_inside_word = set(m_eval_chunks_wordlevel_null_segmod[w_idx])

        #print(sorted(gold_chunks_inside_word))
        #print(sorted(eval_chunks_inside_word))

        # handle above assert case properly
        if len(m_eval_chunks_wordlevel_null_segmod) > w_idx:
            eval_chunks_inside_word = set(m_eval_chunks_wordlevel_null_segmod[w_idx])
        else:
            eval_chunks_inside_word = set() # take everything as wrong after a certain point, if word spacing got screwed up

        correct_chunks_inside_word = gold_chunks_inside_word & eval_chunks_inside_word
        all_correct_m_chunks_wordlevel_null_segmod += len(correct_chunks_inside_word)
        all_total_m_chunks_wordlevel_null_segmod += len(gold_chunks_inside_word)





    # evaluate based on NULL mod morph word-level chunk accuracy (look at each word)
    for w_idx in range(len(m_gold_chunks_wordlevel)):
        # compare inner word chunks
        gold_chunks_inside_word = set(m_gold_chunks_wordlevel[w_idx])
        eval_chunks_inside_word = set(m_eval_chunks_wordlevel_null_mod[w_idx])

        #print(sorted(gold_chunks_inside_word))
        #print(sorted(eval_chunks_inside_word))

        # handle above assert case properly
        if len(m_eval_chunks_wordlevel_null_mod) > w_idx:
            eval_chunks_inside_word = set(m_eval_chunks_wordlevel_null_mod[w_idx])
        else:
            eval_chunks_inside_word = set() # take everything as wrong after a certain point, if word spacing got screwed up

        correct_chunks_inside_word = gold_chunks_inside_word & eval_chunks_inside_word
        all_correct_m_chunks_wordlevel_null_mod += len(correct_chunks_inside_word)
        all_total_m_chunks_wordlevel_null_mod += len(gold_chunks_inside_word)



    # evaluate based on morph word-level chunk accuracy (look at each word)
    for w_idx in range(len(m_gold_chunks_wordlevel)):
        is_oov_word = (morph_only_fromword(m_gold_chunks_wordlevel[w_idx]) in wordlevel_oov_token_set)

        # compare inner word chunks
        gold_chunks_inside_word = set(m_gold_chunks_wordlevel[w_idx])
        eval_chunks_inside_word = set(m_eval_chunks_wordlevel[w_idx])

        # handle above assert case properly
        if len(m_eval_chunks_wordlevel) > w_idx:
            eval_chunks_inside_word = set(m_eval_chunks_wordlevel[w_idx])
        else:
            eval_chunks_inside_word = set() # take everything as wrong after a certain point, if word spacing got screwed up

        correct_chunks_inside_word = gold_chunks_inside_word & eval_chunks_inside_word
        all_correct_m_chunks_wordlevel += len(correct_chunks_inside_word)
        all_total_m_chunks_wordlevel += len(gold_chunks_inside_word)

        # increment the OOV metric if applicable as well
        if is_oov_word:
            all_correct_m_chunks_wordlevel_oov += len(correct_chunks_inside_word)
            all_total_m_chunks_wordlevel_oov += len(gold_chunks_inside_word)

    sentence_is_correct = True

    # evaluate based on tag word-level chunk accuracy (look at each word)
    for w_idx in range(len(t_gold_chunks_wordlevel)):
        is_oov_word = (morph_only_fromword(m_gold_chunks_wordlevel[w_idx]) in wordlevel_oov_token_set)
        # print(m_gold_chunks_wordlevel[w_idx], t_gold_chunks_wordlevel[w_idx])
        # is_oov_word needs to check MORPH input. combined input may not be same
        '''
            MORPH: 이러한
            TAG:   이러하ㄴ

        '''

        # compare inner word chunks
        gold_chunks_inside_word = set(t_gold_chunks_wordlevel[w_idx])

        if len(t_eval_chunks_wordlevel) > w_idx:
            eval_chunks_inside_word = set(t_eval_chunks_wordlevel[w_idx])
        else:
            eval_chunks_inside_word = set() # take everything as wrong after a certain point, if word spacing got screwed up

        correct_chunks_inside_word = gold_chunks_inside_word & eval_chunks_inside_word

        if len(correct_chunks_inside_word) != len(gold_chunks_inside_word):
            sentence_is_correct = False
        else:
            all_correct_words += 1
            if is_oov_word:
                all_correct_words_oov += 1

        all_total_words += 1
        if is_oov_word:
            all_total_words_oov += 1

        all_correct_e2e_chunks_wordlevel += len(correct_chunks_inside_word)
        all_total_e2e_chunks_wordlevel += len(gold_chunks_inside_word)

        # increment the OOV metric if applicable as well
        if is_oov_word:
            all_correct_e2e_chunks_wordlevel_oov += len(correct_chunks_inside_word)
            all_total_e2e_chunks_wordlevel_oov += len(gold_chunks_inside_word)


    correct_m_chunks = m_gold_chunks & m_eval_chunks
    total_m_chunks = len(m_gold_chunks)

    all_correct_m_chunks += len(correct_m_chunks)
    all_total_m_chunks += total_m_chunks


    correct_e2e_chunks = t_gold_chunks & t_eval_chunks
    total_e2e_chunks = len(t_gold_chunks)

    if len(correct_e2e_chunks) < total_e2e_chunks:
        #print('Gold', sorted(t_gold_chunks))
        #print('Eval', sorted(t_eval_chunks))
        #print('Invalid:', sorted(t_gold_chunks - t_eval_chunks))

        ## FIXME: possible that mode2 input has OOV created by mode0
        # in that case mode2 chunk might be something weird? or not in new one ?
        # because a deepcopy is done of gold and then only labels are changed
        # in predict function, even if the OOV unit isn't in the tag stage's
        # word2index (lexicon) it doesn't get changed.

        '''
        # take whatever was from mode0out
        print('OOV', pred[g_i], '==>', (mode0out[g_i][0], pred[g_i][1]))
        pred[g_i] = (mode0out[g_i][0], pred[g_i][1])
        '''

    all_correct_e2e_chunks += len(correct_e2e_chunks)
    all_total_e2e_chunks += total_e2e_chunks

    if sentence_is_correct:
        all_correct_sentences += 1
    all_total_sentences += 1

    #break

# sentlevel includes proper space detection so that's why the chunk count is higher
print('morph-sentlevel: %d/%d (%.2f%%)' % (all_correct_m_chunks, all_total_m_chunks, 100.0*float(all_correct_m_chunks)/float(all_total_m_chunks)))
print('e2e-sentlevel: %d/%d (%.2f%%)' % (all_correct_e2e_chunks, all_total_e2e_chunks, 100.0*float(all_correct_e2e_chunks)/float(all_total_e2e_chunks)))
print('morph-wordlevel: %d/%d (%.2f%%)' % (all_correct_m_chunks_wordlevel, all_total_m_chunks_wordlevel, 100.0*float(all_correct_m_chunks_wordlevel)/float(all_total_m_chunks_wordlevel)))
print('morph-wordlevel-oov: %d/%d (%.2f%%)' % (all_correct_m_chunks_wordlevel_oov, all_total_m_chunks_wordlevel_oov, 100.0*float(all_correct_m_chunks_wordlevel_oov)/float(all_total_m_chunks_wordlevel_oov)))
print('morph-wordlevel-null-segmod: %d/%d (%.2f%%)' % (all_correct_m_chunks_wordlevel_null_segmod, all_total_m_chunks_wordlevel_null_segmod, 100.0*float(all_correct_m_chunks_wordlevel_null_segmod)/float(all_total_m_chunks_wordlevel_null_segmod)))
print('morph-wordlevel-null-mod: %d/%d (%.2f%%)' % (all_correct_m_chunks_wordlevel_null_mod, all_total_m_chunks_wordlevel_null_mod, 100.0*float(all_correct_m_chunks_wordlevel_null_mod)/float(all_total_m_chunks_wordlevel_null_mod)))
print('e2e-wordlevel: %d/%d (%.2f%%)' % (all_correct_e2e_chunks_wordlevel, all_total_e2e_chunks_wordlevel, 100.0*float(all_correct_e2e_chunks_wordlevel)/float(all_total_e2e_chunks_wordlevel)))
print('e2e-wordlevel-oov: %d/%d (%.2f%%)' % (all_correct_e2e_chunks_wordlevel_oov, all_total_e2e_chunks_wordlevel_oov, 100.0*float(all_correct_e2e_chunks_wordlevel_oov)/float(all_total_e2e_chunks_wordlevel_oov)))
print('correct words (all e2e chunks correct): %d/%d (%.2f%%)' % (all_correct_words, all_total_words, 100.0*float(all_correct_words)/float(all_total_words)))
print('correct oov words (all e2e chunks correct): %d/%d (%.2f%%)' % (all_correct_words_oov, all_total_words_oov, 100.0*float(all_correct_words_oov)/float(all_total_words_oov)))
print('correct sentences (all e2e words correct): %d/%d (%.2f%%)' % (all_correct_sentences, all_total_sentences, 100.0*float(all_correct_sentences)/float(all_total_sentences)))


'''
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
'''
