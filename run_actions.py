# runs the following actions specified in the stage
# involves some legacy code from previous project

import sys
import argparse
import korean_morph_seq2
from bio_tools import *
from logging import handlers

parser = argparse.ArgumentParser(description='Runs actions output by a stage and re-outputs BIO-format file.')

# only morph for now
#parser.add_argument('model_type', type=str,
#                    help='morph or tag')
parser.add_argument('morph_eval', type=str, default=None,
                    help='Path to load inference file from')
parser.add_argument('morph_eval_out', type=str, default=None,
                    help='Path at which to store BIO file with action output (same number of sentences as input)')
args = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : [%(name)s] : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('RunActions')

#assert args.model_type == 'morph'

'''
Original oracle code works in terms of tuple

Converts our action string to a tuple
'''
def restore_original_action_tuple(action_str):
    if action_str == 'B-KEEP' or action_str == 'I-KEEP' or action_str == 'NOOP':
        return (action_str,)
    elif action_str.startswith('MOD'):
        return ('MOD', eval(action_str.split('MOD:', 1)[1]))
    else:
        assert None, 'unknown action: ' + action_str


logger.info('Morph inference input: %s' % args.morph_eval)
logger.info('Morph action output: %s' % args.morph_eval_out)
morph_eval = BIODataInput(args.morph_eval)

sentence_count = len(morph_eval.sentences)

fd = open(args.morph_eval_out, 'w', encoding='utf-8')
wrote_first_para = False

for i in range(sentence_count):
    m_eval = morph_eval.sentences[i]
    m_eval_chunks = m_eval.get_label_chunks()

    unit_words = [inp[0] for inp in m_eval.inputs]
    action_tuples = list(map(restore_original_action_tuple, m_eval.labels))

    logger.debug('Unit words: %s' % unit_words)
    logger.debug('Action tuples: %s' % action_tuples)

    segments = korean_morph_seq2.restoreOrigSegments(action_tuples, unit_words)

    logger.debug('Segment output: %s' % segments)

    new_sentence = BIODataSentence()
    for s in segments:
        new_sentence.inputs.append(tuple([s]))
        new_sentence.labels.append('NULL') # to be filled in by tagging stage

    if wrote_first_para:
        fd.write('\n\n')
    else:
        wrote_first_para = True

    fd.write(str(new_sentence))

fd.close()
