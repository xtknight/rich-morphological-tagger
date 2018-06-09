#
# detect commonalities among specified suffixes in the CoNLL and add morpheme
# for lemma/POSTAG + suffix/MORPHTAG
# (similar to Sejong corpus)
#
import sys
from conll_utils import *

all_suffixes = set()

with open('tr-suffixes', 'r', encoding='utf-8') as fd:
    suffixes = fd.read()

for s in suffixes.split('\n'):
    s = s.strip()
    if not s:
        continue
    if s.startswith('-'): # TODO: fix this later
        s = s[1:]
    s = s.strip()
    if not s:
        continue
    all_suffixes.add(s)

all_suffixes = sorted(all_suffixes)

print('all_suffixes', all_suffixes)

# morphological properties that occurred
# in each specified suffix
suffix_morph_list = {}

for s in all_suffixes:
    suffix_morph_list[s] = {}


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
            if wd.FEATS:
                for s in all_suffixes:
                    if wd.FORM.endswith(s):
                        #print(wd.FORM, s, sorted(wd.FEATS))
                        for f in wd.FEATS:
                            #print('co-occur', s, f)
                            # update co-occurrence matrix
                            if f not in suffix_morph_list[s]:
                                suffix_morph_list[s][f] = 0
                            suffix_morph_list[s][f] += 1

    # pre-scan for errors
    #for sentence in corpus:
    #    error_found = False
    #    for wd in sentence.tokens:
    #        if wd.FORM:

procFile('/mnt/deeplearn/CoNLL_2017_SharedTask/ud-treebanks-conll2017/UD_Turkish/tr-ud-traindevtest.conllu')
#print('ları co-occurrences', suffix_morph_list['ları'])

for (k, v) in sorted(suffix_morph_list.items()):
    print('%s co-occurrences: %s' % (k, v))

# convert Tense=Fut|Tense=Past ==>
# {'Tense': ['Fut', 'Past']}
def convertPropListToTable(propList):
    propTable = {}
    for prop in propList:
        if '=' in prop:
            propname = prop.split('=')[0]
            propval = prop.split('=')[1]
            if propname not in propTable:
                propTable[propname] = []
            propTable[propname].append(propval)
        else:
            if prop not in propTable:
                propTable[prop] = []
    return propTable

# now search for properties with unique values in certain suffixes, such as Tense,
# and begin to pull them out
suffix_morph_table = {}
for (k, v) in suffix_morph_list.items():
    suffix_morph_table[k] = convertPropListToTable(v)

for (k, v) in sorted(suffix_morph_table.items()):
    for prop_name, prop_values in v.items():
        if len(prop_values) == 1:
            print('LINK', k, prop_name+'='+prop_values[0])
            print('... co-occurrences', suffix_morph_list[k][prop_name+'='+prop_values[0]])
        elif len(prop_values) == 0:
            print('LINK', k, prop_name)
            print('... co-occurrences', suffix_morph_list[k][prop_name])
        #else:
        #    print('NO_LINK', k, prop_name, prop_values)
        #print()
    print()
