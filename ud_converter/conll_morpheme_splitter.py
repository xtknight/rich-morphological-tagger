#
# detect commonalities among specified suffixes in the CoNLL and add morpheme
# for lemma/POSTAG + suffix/MORPHTAG
# (similar to Sejong corpus)
#
import sys
from conll_utils import *
from simple_ngram_oracle import get_longest_split_pieces, split_at, varlen_oracle

all_suffixes = set()

with open('tr-suffixes', 'r', encoding='utf-8') as fd:
    suffixes = fd.read()

for s in suffixes.split('\n'):
    s = s.strip()
    if not s:
        continue
    if s.startswith('#'):
        continue
    if s.startswith('-'): # TODO: fix this later
        s = s[1:]
    s = s.strip()
    if not s:
        continue
    all_suffixes.add(s)

# for consistency:
# sort by alphabetical order
all_suffixes = sorted(all_suffixes)
# then sort by longest first (stable sort, so alphabetical order should be maintained)
all_suffixes = sorted(all_suffixes, key=len, reverse=True)

print('all_suffixes', all_suffixes)

# morphological properties that occurred
# in each specified suffix
suffix_morph_list = {}

for s in all_suffixes:
    suffix_morph_list[s] = {}


def getSuffix(FORM):
	for s in all_suffixes:
		if FORM.endswith(s):
			return s
	return None

def removeSuffix(FORM, suffix):
    assert len(FORM) >= 1
    assert len(suffix) >= 1
    tmp = FORM[:-len(suffix)]
    return tmp

def procFile(fname):
    global sent_error_count

    fd = open(fname, 'r', encoding='utf-8') #, errors='ignore')
    contents = fd.read()
    fd.close()

    retval_sentences = []

    corpus = ConllFile(keepMalformed=True,
                               checkParserConformity=False,
                               projectivize=False,
                               enableLemmaMorphemes=False,
                               enableLemmaAsStem=True, # UD style
                               compatibleJamo=True)
    corpus.read(contents)

    for sentence in corpus:
        for wd in sentence.tokens:
            #if wd.FEATS:
            print('Orig word:', wd.FORM)

            #suffixes_stripped = wd.FORM
            #while getSuffix(suffixes_stripped):
            #    suffixes_stripped=removeSuffix(suffixes_stripped, getSuffix(suffixes_stripped))

            split_indices = get_longest_split_pieces(list(wd.FORM), all_suffixes)
            #if wd.FORM == 'Geleceğin':
                # debug...
                #print('split_indices', split_indices)
                #sys.exit(1)
                #break

            FORM_split = split_at(list(wd.FORM), split_indices)
            LEMMA_split = split_at(list(wd.LEMMA), split_indices)

            #print('Without suffixes:', suffixes_stripped)
            print('FORM split:', FORM_split)
            print('LEMMA split:', LEMMA_split)
            print('Real lemma:', wd.LEMMA)
            print('FORM->LEMMA oracle actions:', varlen_oracle(FORM_split, LEMMA_split))
            print()

            '''
            if wd.FORM.endswith(s):
                #print(wd.FORM, s, sorted(wd.FEATS))
                for f in wd.FEATS:
                    #print('co-occur', s, f)
                    # update co-occurrence matrix
                    if f not in suffix_morph_list[s]:
                        suffix_morph_list[s][f] = 0
                    suffix_morph_list[s][f] += 1
            '''

procFile('/mnt/deeplearn/CoNLL_2017_SharedTask/ud-treebanks-conll2017/UD_Turkish/tr-ud-traindevtest.conllu')

#7       göstermeyi      göster  VERB    Verb    Aspect=Perf|Case=Acc|Mood=Ind|Polarity=Pos|Tense=Pres|VerbForm=Vnoun    6

