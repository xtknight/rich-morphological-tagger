import sys

# korean extensions
from jamo import h2j, j2hcj
from ext_korean import *

# all 1to2 actions
reservedKeywords = [
    '__REMOVE_FINAL_JAMO_PT1__',
    '__REMOVE_FINAL_JAMO_PT2__',
    '__DECOMPRESS_VOWEL_PT1__',
    '__DECOMPRESS_VOWEL_PT2__',
    '__DUPLICATE_VOWEL_PT1__',
    '__DUPLICATE_VOWEL_PT2__',
    '__R_IRREGULAR_OP_PT1__',
    '__R_IRREGULAR_OP_PT2__',
    '__EU_IRREGULAR_OP_PT1__',
    '__EU_IRREGULAR_OP_PT2__',
    '__H_IRREGULAR_OP_PT1__',
    '__H_IRREGULAR_OP_PT2__',
    '__AE_IRREGULAR_OP_PT1__',
    '__AE_IRREGULAR_OP_PT2__'
]

def greedyOracle(input, lemmaStack):
    for p in lemmaStack:
        assert len(p) > 0, 'zero-length lemma piece given'

    futureLemma = ''.join(lemmaStack)
    actionList = []

    indexMap = dict()
    for i in range(len(input)):
        indexMap[i] = None

    # look for matches from end

    # take care when using these indices
    # these indices are EXCLUSIVE ranges
    # len(...) means nothing matched in end
    # -1 means nothing matched in start

    # first matched character index in input from end
    inputEndIndex = len(input)
    # first matched character index in lemma from end
    lemmaEndIndex = len(futureLemma)

    keepEndCount = 0

    while inputEndIndex > 0 and lemmaEndIndex > 0:
        if input[inputEndIndex-1] == futureLemma[lemmaEndIndex-1]:
            #print('EndCheck', input[inputEndIndex-1], futureLemma[lemmaEndIndex-1])
            indexMap[inputEndIndex-1] = 'KEEP-END'
            lemmaEndIndex -= 1
            inputEndIndex -= 1
            keepEndCount += 1
        else:
            break

    assert inputEndIndex >= 0
    assert lemmaEndIndex >= 0

    #print('lemmaEndIndex', lemmaEndIndex)
    #print('inputEndIndex', inputEndIndex)

    # look for matches from start
    # last matched character index in input from start
    inputStartIndex = -1
    # last matched character index in lemma from start
    lemmaStartIndex = -1

    while inputStartIndex+1 < inputEndIndex and lemmaStartIndex+1 < lemmaEndIndex:
        if input[inputStartIndex+1] == futureLemma[lemmaStartIndex+1]:
            #print('StartCheck', input[inputStartIndex+1], futureLemma[lemmaStartIndex+1])
            indexMap[inputStartIndex+1] = 'KEEP-START'
            lemmaStartIndex += 1
            inputStartIndex += 1
        else:
            break

    #print('lemmaStartIndex', lemmaStartIndex)
    #print('inputStartIndex', inputStartIndex)

    # after we've covered the end,
    # figure out if we're missing anything at the front

    '''
    if len(futureLemma) > len(input):
        # may need to add more in middle or beginning portion.
        # and so in that case, we have to include lemmaEndIndex+1 as well
        # because we are technically adding to an existing operation that would
        # have handled lemmaEndIndex+1
        #assert indexMap[inputStartIndex+1] == 'KEEP-START' or indexMap[inputStartIndex+1] == 'KEEP-END'
        if indexMap[inputStartIndex+1] == 'KEEP-START' or indexMap[inputStartIndex+1] == 'KEEP-END':
            indexMap[inputStartIndex+1] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex+1]
        else:
            # not a KEEP op, so we don't need to add lemmaEndIndex+1
            indexMap[inputStartIndex+1] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex]
    '''

    modAdded = False

    #print(len(futureLemma), len(input))

    if len(futureLemma) > len(input):
        # may need to add more in middle or beginning portion.
        # and so in that case, we have to include lemmaEndIndex+1 as well
        # because we are technically adding to an existing operation that would
        # have handled lemmaEndIndex+1
        #assert indexMap[inputStartIndex+1] == 'KEEP-START' or indexMap[inputStartIndex+1] == 'KEEP-END'
        #print('HANDLE1: need to add...')

        #if lemmaIndex < inputEndIndex
        # if no keep-ends and we have only keep-starts, the burden is all on us to put the whole string in the
        # first available position!!!!

        if keepEndCount == 0:
            #print('XXXX')
            # have to jam it in the last keep-start position (if any exists)
            lastKeepStartPos = max(0, inputStartIndex)

            if indexMap[lastKeepStartPos] == 'KEEP-START':
                #print('CASE1')
                if lastKeepStartPos+1 in indexMap:
                    assert indexMap[lastKeepStartPos+1] == None # ???? shouldn't have anything here
                    # no need to disturb original KEEP-START in this position because we happen to have some extra space ('None's)
                    indexMap[lastKeepStartPos+1] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex+1]
                else:
                    indexMap[lastKeepStartPos] = 'MOD:' + input[inputStartIndex]+futureLemma[lemmaStartIndex+1:lemmaEndIndex+1]
            else:
                #print('CASE2')
                # not a KEEP op, so we don't need to add lemmaEndIndex+1
                ## TODO: correct???
                #print('TODO: correct??')
                # nothing kept at start, so only add lemma stuff?
                indexMap[lastKeepStartPos] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex]
        else:
            if indexMap[inputStartIndex+1] == 'KEEP-START' or indexMap[inputStartIndex+1] == 'KEEP-END':
                ## FIXME: if KEEP-END, still using inputStartIndex??
                indexMap[inputStartIndex+1] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex+1]
            else:
                # not a KEEP op, so we don't need to add lemmaEndIndex+1
                indexMap[inputStartIndex+1] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex]
        #else:
        #    # need to add to absolute first one!!!
        #    if indexMap[0] == 'KEEP-START' or indexMap[0] == 'KEEP-END':
        #        indexMap[0] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex+1]
        #    else:
        #        # not a KEEP op, so we don't need to add lemmaEndIndex+1
        #        indexMap[0] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex]

        modAdded = True

    #for i in range(len(input)):
    #    print('Before IndexMap[%d(%s)] = %s' % (i, input[i], indexMap[i]))

    if not modAdded:
        for i in range(len(input)):
            #print('CHECK', i, input[i], indexMap[i])
            #print('remaining', futureLemma[lemmaStartIndex+1:lemmaEndIndex])

            if indexMap[i] == None:
                #print('... REPLACE must happen here')
                #if i <= lemmaStartIndex:
                #    # we still have useless input to use up
                #    # use it up first
                #    print('use up useless input')
                #    indexMap[i] = 'NOOP'
                #else:
                # we must insert the remaining action here and follow it by NOOPS

                #print('dbg', lemmaStartIndex+1, lemmaEndIndex)

                if lemmaStartIndex+1 == lemmaEndIndex:
                    # but if we have no characters left, then it should just be a NOOP!
                    # otherwise it starts becoming MOD:'', which is not good
                    indexMap[i] = 'NOOP'
                else:
                    assert lemmaStartIndex+1 < lemmaEndIndex
                    indexMap[i] = 'MOD:' + futureLemma[lemmaStartIndex+1:lemmaEndIndex]

                #print('...', indexMap[i])

                # this position can handle one char, but we need n-1 more NOOPS,
                # where n is our expansion
                #noopCount = len(futureLemma[lemmaStartIndex+1:lemmaEndIndex])-1
                #for k in range(i+1, i+1+noopCount):
                #    indexMap[k] = 'NOOP'
                modAdded = True
                break

    if modAdded:
        # now do NOOPS
        for i in range(len(input)):
            if indexMap[i] == None:
                indexMap[i] = 'NOOP'

    #for i in range(len(input)):
    #    print('After IndexMap[%d(%s)] = %s' % (i, input[i], indexMap[i]))

    for i in range(len(input)):
        if indexMap[i].startswith('MOD:'):
            actionList.append(('MOD', indexMap[i][4:]))
        elif indexMap[i] == 'NOOP':
            actionList.append(('NOOP', ))
        else:
            actionList.append(('KEEP', input[i]))

    #print('ORIGINAL_actionList', actionList)

    # python3 convert_from_sejong_balanced_to_segposcorpus.py "/mnt/deeplearn/corpus/balanced/말뭉치/색인/Example_Files/Written/Art/"
    # with redistributeMods2: 1128 unique tags needed
    # with koreanSpecificOpt1: 868 unique tags needed
    # with koreanSpecificOpt2: 818 unique tags needed
    # with koreanSpecificOpt2+PT2: 815 unique tags needed
    # with koreanSpecificOpt2+PT2+Opt1PT2Addition: 812 unique tags needed
    # with koreanSpecificOpt3: 788 unique tags needed
    # with koreanSpecificOpt4: 712 unique tags needed

    #return actionList
    #return redistributeMods2(redistributeMods(fillNoops(actionList, input), input), input)
    #return koreanSpecificOpt4(koreanSpecificOpt3(koreanSpecificOpt2(koreanSpecificOpt1(redistributeMods2(redistributeMods(fillNoops(actionList, input), input), input), input), input), input), input)
    return actionList
    #return fillNoops(actionList, input)
    #return redistributeMods2(redistributeMods(fillNoops(actionList, input), input), input)

def addSegmentation(actionList, origInput, pieces):
    assert len(origInput) == len(actionList), 'must be seq2seq encoding'
    newActionList = []

    charsConsumed = 0
    currentPiece = 0
    currentProcessedPieceLength = 0

    for p in pieces:
        assert len(p) > 0, 'zero-length lemma piece given'

    for i in range(len(actionList)):
        #print('currentPiece', currentPiece)
        if actionList[i][0] == 'KEEP':
            #print('KEEP')
            currentProcessedPieceLength += 1
            #print('A currentProcessedPieceLength', currentProcessedPieceLength)
            #print('A len(pieces[currentPiece])', len(pieces[currentPiece]))
            assert currentProcessedPieceLength <= len(pieces[currentPiece])

            # unnecessary to store contents of all KEEP actions
            # drastically reduces number of actions needed

            if currentProcessedPieceLength == len(pieces[currentPiece]):
                if len(pieces[currentPiece]) == 1:
                    #newActionList.append(('B-' + actionList[i][0], actionList[i][1]))
                    newActionList.append((('B-' + actionList[i][0]), ))
                else:
                    #newActionList.append(('I-' + actionList[i][0], actionList[i][1]))
                    newActionList.append((('I-' + actionList[i][0]), ))

                currentPiece += 1
                currentProcessedPieceLength = 0
            else:
                if currentProcessedPieceLength == 1:
                    #newActionList.append(('B-' + actionList[i][0], actionList[i][1]))
                    newActionList.append((('B-' + actionList[i][0]), ))
                else:
                    #newActionList.append(('I-' + actionList[i][0], actionList[i][1]))
                    newActionList.append((('I-' + actionList[i][0]), ))

            charsConsumed += 1
        elif actionList[i][0] == 'MOD':
            #print('MOD')
            #print('P', actionList[i])
            #assert len(actionList[i][1]) <= 1 # must be nothing or only one char??
            charsConsumed += len(actionList[i][1])
            ## TODO: let's make this a max of one char!!

            expSource = origInput[i]
            expResult = actionList[i][1]

            subList = []

            for c in expResult:
                currentProcessedPieceLength += 1

                #print('B currentProcessedPieceLength', currentProcessedPieceLength)
                #print('B len(pieces[currentPiece])', len(pieces[currentPiece]))
                assert currentProcessedPieceLength <= len(pieces[currentPiece])

                if currentProcessedPieceLength == len(pieces[currentPiece]):
                    if len(pieces[currentPiece]) == 1:
                        subList.append(('B-' + c))
                    else:
                        subList.append(('I-' + c))

                    currentPiece += 1
                    currentProcessedPieceLength = 0
                else:
                    if currentProcessedPieceLength == 1:
                        subList.append(('B-' + c))
                    else:
                        subList.append(('I-' + c))

                charsConsumed += 1

            newActionList.append((actionList[i][0], subList))
        elif actionList[i][0] == 'NOOP':
            newActionList.append(('NOOP', ))
        else:
            assert None, 'unknown action sequence: ' + str(actionList)

    return newActionList

'''
Get the index of the word within the unit list,
given a list of word breaklevels for each unit
'''
def getWordIdx(origInput, origInputUnitIdx, breaklevelsList):
    word_idx = -1
    for unit, lvl in enumerate(breaklevelsList):
        if lvl == 'CHR-WD-BEGIN':
            word_idx += 1
        # word_idx == 0 for first word
        if unit == origInputUnitIdx:
            return word_idx
    assert None, 'word not found'

def getWordCount(breaklevelsList):
    word_cnt = 0
    for lvl in breaklevelsList:
        if lvl == 'CHR-WD-BEGIN':
            word_cnt += 1
    return word_cnt

def restoreOrigSegments(actionList, origInput, breaklevelsList=None):
    assert len(origInput) == len(actionList), 'must be seq2seq encoding'
    if breaklevelsList != None:
        assert len(origInput) == len(breaklevelsList), 'must be one breaklevel per input element'
    output = []
    currentSubwd = ''

    if breaklevelsList != None:
        outputPerWord = [[]]
        currentWordIndex = 0

    for a_i in range(len(actionList)):
        a = actionList[a_i]

        if a[0].endswith('KEEP'):
            #assert len(a[1]) == 1

            if breaklevelsList != None and 'I-KEEP' in a[0]:
                ## @@ here we can fix I-KEEP on word boundary
                if breaklevelsList[a_i] == 'CHR-WD-BEGIN':
                    print('WARNING: I-KEEP on word boundary: obvious model misprediction, so fixing to B-KEEP')
                    print('%s -> %s' % (str(a), str(('B-KEEP', ))))
                    a = ('B-KEEP', )

            if 'B-KEEP' in a[0]:
                # begin but add last if needed
                if len(currentSubwd) > 0:
                    if breaklevelsList != None:
                        #print('add to wd %d:' % (len(outputPerWord)-1), currentSubwd)
                        outputPerWord[-1].append(currentSubwd)
                    output.append(currentSubwd)

                currentSubwd = origInput[a_i]
            elif 'I-KEEP' in a[0]:
                currentSubwd += origInput[a_i]
        elif a[0] == 'MOD':
            if breaklevelsList != None and a[1][0].startswith('I-'):
                ## @@ here we can fix I-MOD on word boundary
                if breaklevelsList[a_i] == 'CHR-WD-BEGIN':
                    print('WARNING: I-MOD on word boundary: obvious model misprediction, so fixing to B-MOD')
                    print('%s -> %s' % (a[1][0], 'B-'+a[1][0][2:]))
                    a[1][0] = 'B-'+a[1][0][2:]

            for expPiece in a[1]:
                assert len(expPiece[2:]) == 1 or expPiece[2:] in reservedKeywords

                if expPiece[2:] in reservedKeywords:
                    if expPiece[2:] == '__REMOVE_FINAL_JAMO_PT1__':
                        actualPiece = splitOffFinalJamo(origInput[a_i])[0]
                    elif expPiece[2:] == '__REMOVE_FINAL_JAMO_PT2__':
                        actualPiece = splitOffFinalJamo(origInput[a_i])[1]
                    elif expPiece[2:] == '__DECOMPRESS_VOWEL_PT1__':
                        actualPiece = decompressVowel(origInput[a_i])[0]
                    elif expPiece[2:] == '__DECOMPRESS_VOWEL_PT2__':
                        actualPiece = decompressVowel(origInput[a_i])[1]
                    elif expPiece[2:] == '__DUPLICATE_VOWEL_PT1__':
                        actualPiece = duplicateVowel(origInput[a_i])[0]
                    elif expPiece[2:] == '__DUPLICATE_VOWEL_PT2__':
                        actualPiece = duplicateVowel(origInput[a_i])[1]
                    elif expPiece[2:] == '__R_IRREGULAR_OP_PT1__':
                        actualPiece = R_irregularOperation(origInput[a_i])[0]
                    elif expPiece[2:] == '__R_IRREGULAR_OP_PT2__':
                        actualPiece = R_irregularOperation(origInput[a_i])[1]
                    elif expPiece[2:] == '__EU_IRREGULAR_OP_PT1__':
                        actualPiece = EU_irregularOperation(origInput[a_i])[0]
                    elif expPiece[2:] == '__EU_IRREGULAR_OP_PT2__':
                        actualPiece = EU_irregularOperation(origInput[a_i])[1]
                    elif expPiece[2:] == '__H_IRREGULAR_OP_PT1__':
                        actualPiece = H_irregularOperation(origInput[a_i])[0]
                    elif expPiece[2:] == '__H_IRREGULAR_OP_PT2__':
                        actualPiece = H_irregularOperation(origInput[a_i])[1]
                    elif expPiece[2:] == '__AE_IRREGULAR_OP_PT1__':
                        actualPiece = AE_irregularOperation_1(origInput[a_i])[0]
                    elif expPiece[2:] == '__AE_IRREGULAR_OP_PT2__':
                        actualPiece = AE_irregularOperation_1(origInput[a_i])[1]
                    else:
                        assert None, 'unknown action: ' + str(expPiece[2:])

                    if expPiece.startswith('B-'):
                        # begin but add last if needed
                        if len(currentSubwd) > 0:
                            if breaklevelsList != None:
                                #print('add to wd %d:' % (len(outputPerWord)-1), currentSubwd)
                                outputPerWord[-1].append(currentSubwd)
                            output.append(currentSubwd)
                        currentSubwd = actualPiece
                    elif expPiece.startswith('I-'):
                        currentSubwd += actualPiece
                    else:
                        assert None, 'unknown action: ' + str(expPiece[2:])
                else:
                    if expPiece.startswith('B-'):
                        # begin but add last if needed
                        if len(currentSubwd) > 0:
                            if breaklevelsList != None:
                                #print('add to wd %d:' % (len(outputPerWord)-1), currentSubwd)
                                outputPerWord[-1].append(currentSubwd)
                            output.append(currentSubwd)
                        currentSubwd = expPiece[2:]
                    elif expPiece.startswith('I-'):
                        currentSubwd += expPiece[2:]
        elif a[0] == 'NOOP':
            # ....
            pass

        if breaklevelsList != None:
            currentWordIndex = getWordIdx(origInput, a_i, breaklevelsList)
            if currentWordIndex >= len(outputPerWord):
                #print('char=%s: update currentWordIndex=%d' % (origInput[a_i], currentWordIndex))
                assert len(outputPerWord) == currentWordIndex, 'word index should not have incremented by more than one'
                outputPerWord.append([])

    ## TODO: need this portion?
    if len(currentSubwd) > 0:
        if breaklevelsList != None:
            #print('add to wd %d:' % (len(outputPerWord)-1), currentSubwd)
            outputPerWord[-1].append(currentSubwd)
        output.append(currentSubwd)

    if breaklevelsList != None:
        assert getWordCount(breaklevelsList) == len(outputPerWord)
        return outputPerWord
    else:
        return output

def numConsecutiveNoops(actionList, startIndex):
    num = 0
    for a in actionList[startIndex:]:
        if a[0]=='NOOP':
            num += 1
        else:
            break
    return num

'''
Rebalance MOD actions to reduce total number

TODO: might be able to use smart heuristic to make sensible compression
suggestions

[('B-KEEP',), ('MOD', ['I-기', 'B-어', 'B-지']), ('MOD', ['B-ㄴ']), ('I-KEEP',), ('B-KEEP',)]
['옮기', '어', '지', 'ㄴ다', '.']

['I-기', 'B-어', 'B-지'] + ['B-ㄴ']
=
['I-기', 'B-어'] + ['B-지', 'B-ㄴ']

Can do based on substr frequencies
(most common ones get reduced somehow?)
'''
def redistributeMods(actionList, origStr):
    newActionList = []

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            newActionList.append(a)
        elif a[0]=='MOD':
            numToAdd = 0

            if a_i+1 < len(actionList):
                if len(a[1]) == 3 and actionList[a_i+1][0]=='MOD' and len(actionList[a_i+1][1]) == 1:
                    # common Korean pattern 3-1==>2-2
                    fillSeq = a[1][-1:]
                    #print('fillSeq', fillSeq)
                    a = (a[0], a[1][:-1])
                    actionList[a_i+1] = ('MOD', fillSeq + actionList[a_i+1][1][0])

                    # skip the next actions because we modified them
                    numToAdd = 1

            # add modified or non-modified original MOD action
            newActionList.append(a)

            # add the ones we modified, if necessary
            for k in range(numToAdd):
                newActionList.append(actionList[a_i+1+k])

            a_i += numToAdd
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList

'''
ORACLE [('KEEP', '나'), ('MOD', '아가았'), ('KEEP', '다'), ('KEEP', '.')]
[('B-KEEP',), ('MOD', ['B-아', 'B-가', 'B-았']), ('B-KEEP',), ('B-KEEP',)]
['나', '아', '가', '았', '다', '.']

Instead of first KEEP action, modify it to MOD+B-아  to shorten second MOD
action
'''
def redistributeMods2(actionList, origStr):
    newActionList = []

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            numToAdd = 0

            if a_i+1 < len(actionList):
                if actionList[a_i+1][0]=='MOD' and len(actionList[a_i+1][1]) == 3:
                    # common Korean pattern KEEP+MOD3 => MOD2+MOD2
                    fillSeq = actionList[a_i+1][1][0]

                    # change KEEP to MOD of orig char + first char of second action
                    a = ('MOD', origStr[a_i] + fillSeq)
                    # trim first char off second action
                    actionList[a_i+1] = ('MOD', actionList[a_i+1][1][1:])

                    # skip the next actions because we modified them
                    numToAdd = 1

            # add modified or non-modified original MOD action
            newActionList.append(a)

            # add the ones we modified, if necessary
            for k in range(numToAdd):
                newActionList.append(actionList[a_i+1+k])

            a_i += numToAdd
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList


'''
Redistribute long MOD actions amongst NOOPS
'''
def fillNoops(actionList, origStr):
    newActionList = []

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            newActionList.append(a)
        elif a[0]=='MOD':
            numToAdd = 0
            if len(a[1]) > 1:
                # fill subsequent NOOPS as much as possible
                followingNoopCount = numConsecutiveNoops(actionList, a_i+1)

                if followingNoopCount > 0:
                    # num needed to fill...
                    followingNoopCount = min(followingNoopCount, len(a[1]) - 1)
                    noopFillSeq = a[1][-followingNoopCount:]
                    a = (a[0], a[1][:-followingNoopCount])
                    for k in range(followingNoopCount):
                        if origStr[a_i+1+k] == noopFillSeq[k]:
                            # FIXME: is this even possible?
                            # surprise! can just use KEEP
                            actionList[a_i+1+k] = ('KEEP', )
                        else:
                            actionList[a_i+1+k] = ('MOD', noopFillSeq[k])
                    # skip the next actions because we modified them
                    numToAdd = followingNoopCount

            # add modified or non-modified original MOD action
            newActionList.append(a)

            # add the ones we modified, if necessary
            for k in range(numToAdd):
                newActionList.append(actionList[a_i+1+k])

            a_i += numToAdd
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList


'''
Perform Korean-specific operation
(replace MODs with simpler pieces)

진 CHR ('MOD', ['지', 'ㄴ'])
    ==> 1TO2_SEPARATE_FINAL_JAMO
'''
def koreanSpecificOpt1(actionList, origStr):
    #origStr = normalizeToCompatJamo(origStr)
    newActionList = []

    #print('actionList', actionList)

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            newActionList.append(a)
        elif a[0]=='MOD':
            # if list, we already modified, don't modify again
            if type(a[1]) is str and len(a[1]) == 2 and strIsHangulOrJamo(a[1]) and strIsHangul(origStr[a_i]):
                #print('check', splitOffFinalJamo(origStr[a_i]), a[1])

                if splitOffFinalJamo(origStr[a_i]) == a[1]:
                    #print('check2')
                    #a = ('1TO2_SEPARATE_FINAL_JAMO', )
                    # not many suffixes. let's make this a bit simpler
                    # but not completely ruin the opportunity to
                    # tag each split piece also
                    # REMOVE_FINAL_JAMO will be much more common at least
                    # and can be used with B- or I-
                    a = ('MOD', ['__REMOVE_FINAL_JAMO_PT1__', '__REMOVE_FINAL_JAMO_PT2__'])

            # add modified or non-modified original MOD action
            newActionList.append(a)
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList

'''
Vowel decompression support:

쳐 CHR ('MOD', ['치', '어'])
'''
def koreanSpecificOpt2(actionList, origStr):
    #origStr = normalizeToCompatJamo(origStr)
    newActionList = []

    #print('actionList', actionList)

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            newActionList.append(a)
        elif a[0]=='MOD':
            #print(type(a[1]), a[1])
            #print(type(origStr[a_i]), origStr[a_i])
            # if list, we already modified, don't modify again
            if type(a[1]) is str and len(a[1]) == 2 and strIsHangulOrJamo(a[1]) and strIsHangul(origStr[a_i]):
                #print('check', decompressVowel(origStr[a_i]), a[1])

                if decompressVowel(origStr[a_i]) == a[1]:
                    a = ('MOD', ['__DECOMPRESS_VOWEL_PT1__', '__DECOMPRESS_VOWEL_PT2__'])

            # add modified or non-modified original MOD action
            newActionList.append(a)
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList

'''
Vowel duplication support:

잤 CHR ('MOD', ['자', '았'])
'''
def koreanSpecificOpt3(actionList, origStr):
    #origStr = normalizeToCompatJamo(origStr)
    newActionList = []

    #print('actionList', actionList)

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            newActionList.append(a)
        elif a[0]=='MOD':
            #print(type(a[1]), a[1])
            #print(type(origStr[a_i]), origStr[a_i])
            # if list, we already modified, don't modify again
            if type(a[1]) is str and len(a[1]) == 2 and strIsHangulOrJamo(a[1]) and strIsHangul(origStr[a_i]):
                #print('check', duplicateVowel(origStr[a_i]), a[1])

                if duplicateVowel(origStr[a_i]) == a[1]:
                    a = ('MOD', ['__DUPLICATE_VOWEL_PT1__', '__DUPLICATE_VOWEL_PT2__'])

            # add modified or non-modified original MOD action
            newActionList.append(a)
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList

'''
Handle four types of irregular contractions:

('군', '굴ㄴ')
('랬', '렇었')
('빴', '쁘았')
('했', '하았')
'''
def koreanSpecificOpt4(actionList, origStr):
    #origStr = normalizeToCompatJamo(origStr)
    newActionList = []

    #print('actionList', actionList)

    a_i = 0
    while a_i < len(actionList):
        a = actionList[a_i]
        if a[0]=='KEEP':
            newActionList.append(a)
        elif a[0]=='MOD':
            #print(type(a[1]), a[1])
            #print(type(origStr[a_i]), origStr[a_i])
            # if list, we already modified, don't modify again
            if type(a[1]) is str and len(a[1]) == 2 and strIsHangulOrJamo(a[1]) and strIsHangul(origStr[a_i]):
                if R_irregularOperation(origStr[a_i]) == a[1]:
                    a = ('MOD', ['__R_IRREGULAR_OP_PT1__', '__R_IRREGULAR_OP_PT2__'])
                elif EU_irregularOperation(origStr[a_i]) == a[1]:
                    a = ('MOD', ['__EU_IRREGULAR_OP_PT1__', '__EU_IRREGULAR_OP_PT2__'])
                elif H_irregularOperation(origStr[a_i]) == a[1]:
                    a = ('MOD', ['__H_IRREGULAR_OP_PT1__', '__H_IRREGULAR_OP_PT2__'])
                elif AE_irregularOperation_1(origStr[a_i]) == a[1]:
                    a = ('MOD', ['__AE_IRREGULAR_OP_PT1__', '__AE_IRREGULAR_OP_PT2__'])

            # add modified or non-modified original MOD action
            newActionList.append(a)
        else:
            newActionList.append(a)

        a_i += 1

    return newActionList


'''
#greedyOracle('오죽이나', ['오죽이', '이', '나'])
#greedyOracle('이건', ['이', '건'])
#greedyOracle('이건', ['이', '이', '건'])
#greedyOracle('이건', ['이', '이', '이', '건'])
#greedyOracle('아이건', ['아', '이', '이', '이', '건'])
#greedyOracle('아이건', ['아', '이', '이', '건'])
#greedyOracle('아이건', ['아', '이', '건'])

#sys.exit(1)

a = greedyOracle('세계적인', ['세계', '적', '이', 'ㄴ'])
print(a)
b = addSegmentation(a, '세계적인', ['세계', '적', '이', 'ㄴ'])
print(b)
c = restoreOrigSegments(b, '세계적인')
print(c)
assert c == ['세계', '적', '이', 'ㄴ']


a = greedyOracle('지나서', ['지나', '아서'])
b = addSegmentation(a, '지나서', ['지나', '아서'])
c = restoreOrigSegments(b, '지나서')
print(a)
print(b)
print(c)
assert c == ['지나', '아서']

#print(greedyOracle('아지나서', ['지나', '아성']))

a = greedyOracle('고집세', ['고집', '세'])
b = addSegmentation(a, '고집세', ['고집', '세'])
c = restoreOrigSegments(b, '고집세')
print(a)
print(b)
print(c)
assert c == ['고집', '세']


a = greedyOracle('오죽이나', ['오죽이', '이', '나'])
b = addSegmentation(a, '오죽이나', ['오죽이', '이', '나'])
c = restoreOrigSegments(b, '오죽이나')
print(a)
print(b)
print(c)
assert c == ['오죽이', '이', '나']


a = greedyOracle('이건', ['이', '건'])
b = addSegmentation(a, '이건', ['이', '건'])
c = restoreOrigSegments(b, '이건')
print(a)
print(b)
print(c)
assert c == ['이', '건']


a = greedyOracle('이건', ['이', '이', '건'])
b = addSegmentation(a, '이건', ['이', '이', '건'])
c = restoreOrigSegments(b, '이건')
print(a)
print(b)
print(c)
assert c == ['이', '이', '건']


a = greedyOracle('이건', ['이', '이', '이', '건'])
b = addSegmentation(a, '이건', ['이', '이', '이', '건'])
c = restoreOrigSegments(b, '이건')
print(a)
print(b)
print(c)
assert c == ['이', '이', '이', '건']


a = greedyOracle('ABCDEFG', ['주', '시', '오', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
b = addSegmentation(a, 'ABCDEFG', ['주', '시', '오', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
c = restoreOrigSegments(b, 'ABCDEFG')
print(a)
print(b)
print(c)
assert c == ['주', '시', '오', 'A', 'B', 'C', 'D', 'E', 'F', 'G']



a = greedyOracle('‘이리로', ["'", '이리로'])
b = addSegmentation(a, '‘이리로', ["'", '이리로'])
c = restoreOrigSegments(b, '‘이리로')
print(a)
print(b)
print(c)
assert c == ["'", '이리로']



a = greedyOracle('‘아', ["'", '이', '아'])
b = addSegmentation(a, '‘아', ["'", '이', '아'])
c = restoreOrigSegments(b, '‘아')
print(a)
print(b)
print(c)
assert c == ["'", '이', '아']




a = greedyOracle('‘뭐야?', ["'", '뭣', '이', '야', '?'])
print('TEST', a)
b = addSegmentation(a, '‘뭐야?', ["'", '뭣', '이', '야', '?'])
c = restoreOrigSegments(b, '‘뭐야?')
print(b)
print(c)
assert c == ["'", '뭣', '이', '야', '?']



a = greedyOracle('내', ['내', '어'])
print('TEST', a)
b = addSegmentation(a, '내', ['내', '어'])
c = restoreOrigSegments(b, '내')
print(b)
print(c)
assert c == ['내', '어']


a = greedyOracle('박관호씨(40ㆍ주식회사', ['박관호', '씨', '(', '40', 'ㅋ', 'A', '사'])
print('TEST', a)
b = addSegmentation(a, '박관호씨(40ㆍ주식회사', ['박관호', '씨', '(', '40', 'ㅋ', 'A', '사'])
c = restoreOrigSegments(b, '박관호씨(40ㆍ주식회사')
print(b)
print(c)
assert c == ['박관호', '씨', '(', '40', 'ㅋ', 'A', '사']

a = greedyOracle('ABCDEFG', ['AA', 'BB'])
print('ORACLE', a)
b = addSegmentation(a, 'ABCDEFG', ['AA', 'BB'])
c = restoreOrigSegments(b, 'ABCDEFG')
print(b)
print(c)
assert c == ['AA', 'BB']

a = greedyOracle('옮겨진다.', ['옮기', '어', '지', 'ㄴ다', '.'])
print('ORACLE', a)
b = addSegmentation(a, '옮겨진다.', ['옮기', '어', '지', 'ㄴ다', '.'])
c = restoreOrigSegments(b, '옮겨진다.')
print(b)
print(c)
assert c == ['옮기', '어', '지', 'ㄴ다', '.']

a = greedyOracle('나갔다.', ['나', '아', '가', '았', '다', '.'])
print('ORACLE', a)
b = addSegmentation(a, '나갔다.', ['나', '아', '가', '았', '다', '.'])
c = restoreOrigSegments(b, '나갔다.')
print(b)
print(c)
assert c == ['나', '아', '가', '았', '다', '.']

a = greedyOracle('쳐', ['치', '어'])
print('ORACLE', a)
b = addSegmentation(a, '쳐', ['치', '어'])
c = restoreOrigSegments(b, '쳐')
print(b)
print(c)
assert c == ['치', '어']


a = greedyOracle('잤', ['자', '았'])
print('ORACLE', a)
b = addSegmentation(a, '잤', ['자', '았'])
c = restoreOrigSegments(b, '잤')
print(b)
print(c)
assert c == ['자', '았']

a = greedyOracle('унаследовавший', ['унаследов','ать'])
print('ORACLE', a)
b = addSegmentation(a, 'унаследовавший', ['унаследов','ать'])
c = restoreOrigSegments(b, 'унаследовавший')
print(b)
print(c)
assert c == ['унаследов','ать']

a = greedyOracle('танков', ['т', 'а', 'н', 'к'])
print('ORACLE', a)
b = addSegmentation(a, 'танков', ['т', 'а', 'н', 'к'])
c = restoreOrigSegments(b, 'танков')
print(b)
print(c)
assert c == ['т', 'а', 'н', 'к']

a = greedyOracle('утром', ['у', 'т', 'р', 'о'])
print('ORACLE', a)
b = addSegmentation(a, 'утром', ['у', 'т', 'р', 'о'])
c = restoreOrigSegments(b, 'утром')
print(b)
print(c)
assert c == ['у', 'т', 'р', 'о']

# (MOD, '') vs NOOP
PROCESS танков ['т', 'а', 'н', 'к']
[('KEEP', 'т'), ('KEEP', 'а'), ('KEEP', 'н'), ('KEEP', 'к'), ('MOD', ''), ('NOOP',)]
[('B-KEEP',), ('B-KEEP',), ('B-KEEP',), ('B-KEEP',), ('MOD', []), ('NOOP',)]

PROCESS утром ['у', 'т', 'р', 'о']
[('KEEP', 'у'), ('KEEP', 'т'), ('KEEP', 'р'), ('KEEP', 'о'), ('MOD', '')]
[('B-KEEP',), ('B-KEEP',), ('B-KEEP',), ('B-KEEP',), ('MOD', [])]

action_tuples = [('B-KEEP',), ('I-KEEP',), ('B-KEEP',), ('I-KEEP',), ('MOD', ["B-'"]), ('B-KEEP',), ('I-KEEP',), ('B-KEEP',), ('I-KEEP',), ('B-KEEP',), ('B-KEEP',), ('B-KEEP',), ('I-KEEP',), ('B-KEEP',), ('I-KEEP',), ('B-KEEP',), ('MOD', ["B-'"]), ('MOD', ["B-'"]), ('B-KEEP',), ('I-KEEP',), ('I-KEEP',), ('B-KEEP',), ('B-KEEP',), ('B-KEEP',), ('MOD', ['B-__DECOMPRESS_VOWEL_PT1__', 'B-__DECOMPRESS_VOWEL_PT2__']), ('I-KEEP',), ('B-KEEP',), ('B-KEEP',), ('I-KEEP',), ('B-KEEP',), ('MOD', ["B-'"])]
chars = ['딸', '랑', '딸', '랑', '‘', '집', '세', '녀', '석', '을', '알', '아', '내', '거', '든', '…', '’', '‘', '단', '단', '히', '혼', '을', '내', '줘', '야', '겠', '어', '요', '.', '’']
breaklevels = ['CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER']

print('char', chars[3], 'in word', getWordIdx(chars, 3, breaklevels))
print('char', chars[4], 'in word', getWordIdx(chars, 4, breaklevels))

c = restoreOrigSegments(action_tuples, chars, breaklevels)
print(c)

action_tuples = [('B-KEEP',), ('MOD', ['I-__REMOVE_FINAL_JAMO_PT1__', 'B-__REMOVE_FINAL_JAMO_PT2__']), ('I-KEEP',), ('B-KEEP',), ('MOD', ["B-'"])]
chars = ['신', '난', '다', '.', '’']
breaklevels = ['CHR-WD-BEGIN', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER', 'CHR-WD-INTER']
#per_word_segments ["신나ㄴ다.'"]

c = restoreOrigSegments(action_tuples, chars, breaklevels)
print(c)

'''




a = greedyOracle('internationalization', ['inter', 'nation', 'al', 'ize', 'ation'])
print('ORACLE', a)
b = addSegmentation(a, 'internationalization', ['inter', 'nation', 'al', 'ize', 'ation'])
c = restoreOrigSegments(b, 'internationalization')
print(b)
print(c)
assert c == ['inter', 'nation', 'al', 'ize', 'ation']
