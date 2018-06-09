# convert UD corpus with subtokens to corpus usable by our preprocessor
import sys
import argparse
import logging

parser = argparse.ArgumentParser(description='Process CoNLL-U corpus with subtokens and output one without subtokens')

# Required positional argument
parser.add_argument('input_file', type=str,
                    help='Input CoNLL-U file (Universal Dependencies)')
#parser.add_argument('--right-to-left', action='store_true', default=False,
#                    help='Use when processing RTL language (e.g., Hebrew)')
parser.add_argument('--conjoining', action='store_true', default=False,
                    help='Remove conjoining to help oracle algorithm find common characters (e.g., Arabic)')

args = parser.parse_args()

contents = None
with open(args.input_file, 'r', encoding='utf-8') as fd:
    contents = fd.read()

out_file = open(args.input_file + '.out', 'w', encoding='utf-8')

rangeIDs = set()
outID = 1

rangeBegin = None
rangeEnd = None
rangeFullword = None
rangeMorphs = []

for ln in contents.split('\n'):
    if ln.startswith('#'):
        out_file.write(ln + '\n')
        continue
    ln = ln.strip()
    if not ln:
        # new sentence
        outID = 1
        out_file.write('\n')
        continue
    fields = ln.split('\t')
    ID = fields[0]
    if ID == '1' or ID.startswith('1-'):
        # new sentence
        rangeIDs = set()
    if '-' in ID:
        start, end = ID.split('-')
        start = int(start)
        end = int(end)
        rangeBegin, rangeEnd = start, end
        rangeFullword = fields[1] # ??? right?
        rangeMorphs = []
        #print('make range', rangeBegin, rangeEnd)
        for k in range(start, end+1):
            assert k not in rangeIDs, 'duplicate or overlapping ID found'
        continue
    else:
        assert int(ID) not in rangeIDs
        rangeIDs.add(int(ID))

    #print('process', ID)

    if rangeBegin != None and rangeEnd != None and rangeFullword != None:
        #print('check in range', list(range(rangeBegin,rangeEnd+1)))
        if int(ID) in list(range(rangeBegin,rangeEnd+1)):
            #print('in range:', ID)
            # is this the first ID in the range?
            # if so, output a line at first

            if fields[2] == '_':
                fields[2] = fields[1] # lemma same

            assert fields[3] != '_' # pos tag
            if fields[5] == '_':
                # no morph tag
                morph = fields[2]+('/POS='+fields[3].replace('+','_').replace('/','_')).lower()
            else:
                morph = fields[2]+('/POS='+fields[3].replace('+','_').replace('/','_')+'|'+fields[5].replace('+','_').replace('/','_')).lower()
            #print('add rangemorph', morph)
            rangeMorphs.append(morph)

            #assert fields[1]==fields[2], fields[1]+'!='+fields[2]

            ## TODO: investigate what fields[2] here means

            if int(ID) == rangeBegin:
                #print('at start of range')
                out_file.write('\t'.join([str(outID), rangeFullword]))
                outID += 1
            elif int(ID) == rangeEnd:
                #print('at end of range')
                # add all morphs, if necessary in reverse order if RTL
                #if args.right_to_left:
                #    rangeMorphs.reverse()

                lemma_field = ' + '.join(rangeMorphs)
                #print('out', lemma_field)

                out_file.write('\t' + '\t'.join([lemma_field, '_', '_', '_', '_', '_', '_', '_']) + '\n')

                rangeBegin = None
                rangeEnd = None
                rangeFullword = None
                rangeMorphs = []
    else:
        #print('not in range:', ID)
        # just output as-is but change ID
        #assert fields[1]==fields[2], fields[1]+'!='+fields[2]

        fullword = fields[1]

        if fields[2] == '_':
            fields[2] = fields[1] # lemma same

        assert fields[3] != '_' # pos tag
        if fields[5]=='_':
            # no morph tag
            lemma_field = fields[2]+('/POS='+fields[3].replace('+','_').replace('/','_')).lower()
        else:
            lemma_field = fields[2]+('/POS='+fields[3].replace('+','_').replace('/','_')+'|'+fields[5].replace('+','_').replace('/','_')).lower()
        out_file.write('\t'.join([str(outID), fullword]))
        out_file.write('\t' + '\t'.join([lemma_field, '_', '_', '_', '_', '_', '_', '_']) + '\n')
        #out_file.write('\t'.join(str(outID), fields[1], '_', '_', '_', '_', '_', '_', '_'))
        outID += 1

assert rangeBegin == None
assert rangeEnd == None
assert rangeFullword == None
assert rangeMorphs == []

out_file.close()
