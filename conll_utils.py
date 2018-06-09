'''
A set of classes to handle input and output of CoNLL-U files

http://universaldependencies.org/docs/format.html

The Parsed* classes are useful to store extra properties needed during
the parsing process that are external to the Conll instances themselves
'''

import logging
import well_formed_filter
from jamo import h2j, j2hcj

def isNonCompatibilityJamo(c):
    assert len(c) == 1
    # HANGUL JAMO: (U+1100-U+11FF)
    if ord(c) >= 0x1100 and ord(c) <= 0x11FF:
        return True
    else:
        return False

def normalizeToCompatJamo(s):
    out = ''
    for c in s:
        if isNonCompatibilityJamo(c):
            out += j2hcj(c)
        else:
            out += c
    assert len(s) == len(out)
    return out

def encodeNoneAsUnderscore(s):
    if s == None:
        return '_'
    else:
        return s

def encodeNoneAsUnderscore_Int(i):
    if i == None:
        return '_'
    else:
        return str(i)

'''
Represents a CoNLL token and all its properties (except index)
'''
class ConllToken(object):
    def __init__(self):
        self.FORM = None
        self.LEMMA = None
        self.UPOSTAG = None
        self.XPOSTAG = None
        self.FEATS = []

        '''
        Make sure to subtract one from the HEAD value in the file
        Root becomes -1

        HEAD then becomes n, which refers to the n'th 0-based index entry
        in the parent ConllSentence

        Our parser also requires this to start at -1
        '''
        self.HEAD = None

        self.DEPREL = None
        self.DEPS = None
        self.MISC = None

        # morpheme extraction extension
        self.morphemes = []

    def __str__(self):
        return self.toFileOutput('_')

    def __repr__(self):
        return self.__str__()

    def parseMorphemesFromLemma(self):
        self.morphemes = []
        if self.LEMMA:
            for elem in self.LEMMA.split(' + '):
                word, pos = elem.rsplit('/', 1)
                #pos = 'SEG' # HACK TEST
                self.morphemes.append((word, pos))

    # SPMRL style
    def parseMorphemesFromLemmaAndXPOSTAG(self):
        self.morphemes = []
        if self.LEMMA and self.XPOSTAG:
            assert len(self.LEMMA.split('+')) == len(self.XPOSTAG.split('+'))
            for word, pos in zip(self.LEMMA.split('+'), self.XPOSTAG.split('+')):
                self.morphemes.append((word.strip(), pos.strip().upper()))

    def parseStemFromLemma(self):
        self.morphemes = []
        if self.LEMMA:
            self.morphemes.append((self.LEMMA, 'STEM'))

    def toFileOutput(self, ID):
        def checkTab(s):
            assert '\t' not in s, 'field must not contain a tab: ' + s
            return s

        def checkPipe(s):
            assert '|' not in s, 'field must not contain a pipe: ' + s
            return s

        assert self.FORM != None
        assert type(self.FEATS) is list

        cols = [str(ID),
            checkTab(self.FORM),
            checkTab(encodeNoneAsUnderscore(self.LEMMA)),
            checkTab(encodeNoneAsUnderscore(self.UPOSTAG)),
            checkTab(encodeNoneAsUnderscore(self.XPOSTAG)),
            '|'.join(checkPipe(checkTab(f)) for f in self.FEATS),
            encodeNoneAsUnderscore_Int(self.HEAD+1), # +1 when writing as file
            checkTab(encodeNoneAsUnderscore(self.DEPREL)),
            checkTab(encodeNoneAsUnderscore(self.DEPS)),   # TODO
            checkTab(encodeNoneAsUnderscore(self.MISC))]

        return '\t'.join(cols)

'''
Represents a ConllToken, as parsed
'''
class ParsedConllToken(ConllToken):
    def __init__(self):
        super().__init__()
        self.parsedLabel = None
        self.parsedHead = None
        self.HEAD = -1 # match default value in sentence.proto

    def setParsedLabel(self, label):
        self.parsedLabel = label

    def setParsedHead(self, head):
        self.parsedHead = head

    def clearParsedHead(self):
        self.parsedHead = -1 # match ParserState: always use -1 as <ROOT>

'''
Stores an ordered list of CoNLL tokens
'''
class ConllSentence(object):
    def __init__(self):
        self.tokens = []

    '''
    Convert to file output representation
    '''
    def toFileOutput(self):
        return '\n'.join(self.tokens[ID-1].toFileOutput(ID) \
            for ID in range(1, len(self.tokens)+1))

    def genSyntaxNetJson(self, token, break_level=None, start_index=0):
        break_contents = ''
        if break_level:
            break_contents = \
'''
  break_level       : %s''' % break_level

        return \
'''token: {
  word    : "%s"
  start   : %d
  end     : %d
  head    : %d
  tag     : "%s"
  category: "%s"
  label   : "%s"%s
}''' % (token.FORM, start_index, start_index+len(token.FORM)-1, token.HEAD,
        token.XPOSTAG, token.UPOSTAG, token.DEPREL, break_contents)

    def genSyntaxNetTextHeader(self):
        return 'text       : "%s"' % (' '.join(t.FORM for t in self.tokens))

    '''
    Convert to SyntaxNet JSON format
    '''
    def toSyntaxNetJson(self):
        out = []
        start_index = 0
        out.append(self.genSyntaxNetTextHeader())
        for i in range(len(self.tokens)):
            if i == 0:
                out.append(self.genSyntaxNetJson(self.tokens[i],
                                                 break_level='SENTENCE_BREAK',
                                                 start_index=start_index))
            else:
                out.append(self.genSyntaxNetJson(self.tokens[i],
                                                 start_index=start_index))
            start_index += len(self.tokens[i].FORM) + 1 # assume space
        return '\n'.join(out)

    '''
    Not perfect, but hash helps optimize when finding duplicates
    (probably useful for POS tagging only)
    '''
    def __hash__(self):
        return \
            hash(tuple([t.FORM for t in self.tokens])) * \
            hash(tuple([t.LEMMA for t in self.tokens])) * \
            hash(tuple([t.UPOSTAG for t in self.tokens])) * \
            hash(tuple([t.XPOSTAG for t in self.tokens]))

    def is_equal(self, other):
        self_forms = [t.FORM for t in self.tokens]
        other_forms = [t.FORM for t in other.tokens]

        # quick checks
        if len(self_forms) != len(other_forms):
            return False

        if self_forms != other_forms:
            return False

        self_lemmas = [t.LEMMA for t in self.tokens]
        other_lemmas = [t.LEMMA for t in other.tokens]

        if self_lemmas != other_lemmas:
            return False

        self_upostags = [t.UPOSTAG for t in self.tokens]
        other_upostags = [t.UPOSTAG for t in other.tokens]

        if self_upostags != other_upostags:
            return False

        self_xpostags = [t.XPOSTAG for t in self.tokens]
        other_xpostags = [t.XPOSTAG for t in other.tokens]

        if self_xpostags != other_xpostags:
            return False

        return True

    '''
    Output the token separated by spaces
    '''
    def toSimpleRepresentation(self):
        return ' '.join(t.FORM for t in self.tokens)

class ParsedConllSentence(ConllSentence):
    def __init__(self, docid):
        super().__init__()
        self.docid_ = docid

    def docid(self):
        return self.docid_

    ## checked accessor
    def mutableToken(self, i):
        assert i >= 0
        assert i < len(self.tokens)
        return self.tokens[i]

    def tokenSize(self):
        return len(self.tokens)
    
'''
Stores an ordered list of sentences within a CoNLL file

keepMalformed:
Whether to retain non-projective and invalid examples

projectivize:
Whether to retain non-projective examples by projectivizing them

logStats:
Log statistics about the corpus

enableLemmaMorphemes:
Enable parsing of morphemes in the LEMMA column, format: 웅가로/NNP + 가/JKS

6	웅가로가	웅가로/NNP + 가/JKS	_	_	_	11	NP_SBJ	_	_
'''
class ConllFile(object):
    def __init__(self, parsed=False,
                 checkParserConformity=False,
                 keepMalformed=False, projectivize=False,
                 logStats=False, enableLemmaMorphemes=False,
                 enableLemmaAsStem=False,
                 compatibleJamo=False,
                 fixMorphemeLabelBugs=False,
                 spmrlFormat=False,
                 koreanAsJamoSeq=False):
        #self.sentenceIndex = None
        self.sentences = []
        # use parsed variant of structures
        self.parsed = parsed
        self.logger = logging.getLogger('ConllUtils')
        self.keepMalformed = keepMalformed
        self.projectivize = projectivize
        self.logStats = logStats
        self.enableLemmaMorphemes = enableLemmaMorphemes
        self.enableLemmaAsStem = enableLemmaAsStem

        if self.enableLemmaAsStem:
            assert not self.enableLemmaMorphemes, 'cannot enable both lemma-as-stem and lemma-morphemes'

        self.compatibleJamo = compatibleJamo
        self.checkParserConformity = checkParserConformity
        self.fixMorphemeLabelBugs = fixMorphemeLabelBugs
        self.spmrlFormat = spmrlFormat
        self.koreanAsJamoSeq = koreanAsJamoSeq


    '''
    Read CoNLL-U from the given string

    excludeCols: CoNLL column indices to exclude from reading
                 sometimes we just want to get rid of certain
                 attributes of a token
                 1-based index
    '''
    def read(self, s, excludeCols=[]):
        assert 1 not in excludeCols, 'cannot exclude reading of ID'
        assert 2 not in excludeCols, 'cannot exclude reading of FORM'

        if self.checkParserConformity:
            well_formed_inst = well_formed_filter.WellFormedFilter()
        else:
            assert self.keepMalformed, 'in order to discard malformed ' \
                                       'sentences, you must enable parser ' \
                                       'conformity checking'

        # arbitrary ID that can be used with parser
        if self.parsed:
            docid = 0

        ln_num = 0

        current_sentence = None
        
        # if we encounter an error during processing a sentence
        invalid_sentence = False

        # set up iterator
        # if there is no iterator, set one up
        # if there was an iterator, leave it at its current position
        #if self.sentenceIndex == None:
        #    self.sentenceIndex = len(self.sentences)

        def commit(s):
            # if we're even getting rid of malformed sentences in the first
            # place...
            if self.checkParserConformity:
                if not self.keepMalformed:
                    if not well_formed_inst.isWellFormed(s,
                            projectivize=self.projectivize):
                        # if the sentence is non-projective and projectivize
                        # is enabled, the sentence will be fixed and not discarded
                        self.logger.debug('line %d: discarding malformed or non' \
                            '-projective sentence: "%s"' % \
                            (ln_num, s.toSimpleRepresentation()))
                        # as long as we discard the sentence here,
                        # discarded sentences' words, tags, and labels
                        # won't be added to the lexicon, which is exactly the
                        # behavior we want.
                        return

            self.sentences.append(s)

        def processUnderscore(s):
            if s == '_':
                return None
            else:
                return s

        # token index (to check that it's in order)
        current_ID = 0

        lines = s.split('\n')
        for ln in lines:
            #print('PROC', ln)
            ln_num += 1
            ln = ln.strip()
            if ln.startswith(';'):
                # ignore comments
                continue
            if not ln:
                # a completely blank line indicates we need to commit the
                # current sentence
                if current_sentence != None:
                    if not invalid_sentence:
                        commit(current_sentence)

                    current_sentence = None
                    current_ID = 0

                # 2017/05/25: FIXED: reset invalid sentence state on new sentence
                invalid_sentence = False
                continue
            if ln[0] == '#': # ignore comments completely
                continue
            if invalid_sentence: # don't process invalid sentences
                continue
            cols = [x.strip() for x in ln.split('\t')]
            assert len(cols) >= 2, \
                'line %d: must have at least ID and FORM: ' % ln_num + str(cols)

            if '-' in cols[0] or '.' in cols[0]:
                self.logger.warning('line %d: not implemented: ID=%s, ' \
                                    'invalidating sentence' % (ln_num, cols[0]))
                invalid_sentence = True
                continue
            else:
                ID = int(cols[0])
                assert ID==current_ID+1, 'line %d: token IDs must be in order' \
                   ' and increment by one' % ln_num

            current_ID = ID

            if current_ID == 1:
                if self.parsed:
                    current_sentence = ParsedConllSentence(docid)
                    docid += 1
                else:
                    current_sentence = ConllSentence()

            if self.parsed:
                current_token = ParsedConllToken()
            else:
                current_token = ConllToken()

            #if self.parsed:
            #    current_token.FORM = normalizeDigits(cols[1])
            #else:
            #    current_token.FORM = cols[1]

            # for SyntaxNet,
            # normalization ONLY happens in lexicon builder
            # yet numbers and up as <UNKNOWN> during training
            # interesting...

            # let this be underscore if needed (don't call processUnderscore())
            if self.compatibleJamo:
                cols[1] = normalizeToCompatJamo(cols[1])
            current_token.FORM = cols[1]

            if len(cols) > 2 and (3 not in excludeCols):
                # let this be underscore if needed
                # (don't call processUnderscore())
                if self.compatibleJamo:
                    cols[2] = normalizeToCompatJamo(cols[2])
                current_token.LEMMA = cols[2]

                if self.enableLemmaMorphemes:
                    if self.spmrlFormat:
                        # XPOSTAG needed here
                        if len(cols) > 4 and (5 not in excludeCols):
                            current_token.XPOSTAG = processUnderscore(cols[4])

                        try:
                            current_token.parseMorphemesFromLemmaAndXPOSTAG()
                        except:
                            self.logger.warning('line %d: invalid SPMRL '
                                                'morpheme sequence: %s, %s, '
                                'invalidating sentence' % (ln_num, \
                                current_token.LEMMA,
                                current_token.XPOSTAG))

                            invalid_sentence = True
                            continue
                    else:
                        try:
                            current_token.parseMorphemesFromLemma()
                        except:
                            self.logger.warning('line %d: invalid morpheme '
                                                'sequence: %s,'
                                ' invalidating sentence' % (ln_num, \
                                current_token.LEMMA))

                            invalid_sentence = True
                            continue
                elif self.enableLemmaAsStem:
                    try:
                        current_token.parseStemFromLemma()
                    except:
                        self.logger.warning('line %d: invalid stem: %s,'
                            ' invalidating sentence' % (ln_num, \
                            current_token.LEMMA))

                        invalid_sentence = True
                        continue

                if self.fixMorphemeLabelBugs:
                    if len(current_token.morphemes) > 0 and len(current_token.FORM) > 0:
                        # match case of first char of morpheme with
                        # case of first char of FORM
                        # @@ TODO: also remove + or - within FORM?
                        #if current_token.morphemes[0][0][0]
                        if current_token.FORM[0].islower():
                            current_token.morphemes[0] = (current_token.morphemes[0][0][0].lower() + current_token.morphemes[0][0][1:], current_token.morphemes[0][1])
                        elif current_token.FORM[0].isupper():
                            current_token.morphemes[0] = (current_token.morphemes[0][0][0].upper() + current_token.morphemes[0][0][1:], current_token.morphemes[0][1])

                        #print(current_token.FORM, current_token.morphemes)

                    # also fix LEMMA if necessary
                    if len(current_token.LEMMA) > 0 and len(current_token.FORM) > 0:
                        # match case of first char of morpheme with
                        # case of first char of FORM
                        # @@ TODO: also remove + or - within FORM?
                        #if current_token.morphemes[0][0][0]
                        if current_token.FORM[0].islower():
                            current_token.LEMMA = current_token.LEMMA[0].lower() + current_token.LEMMA[1:]
                        elif current_token.FORM[0].isupper():
                            current_token.LEMMA = current_token.LEMMA[0].upper() + current_token.LEMMA[1:]

                        #print(current_token.FORM, current_token.LEMMA)

            if self.koreanAsJamoSeq:
                # convert FORM, LEMMA, morphemes to all Jamo if input is Hangul
                current_token.FORM = j2hcj(h2j(current_token.FORM))
                current_token.LEMMA = j2hcj(h2j(current_token.LEMMA))
                if current_token.morphemes:
                    for m_idx, m in enumerate(current_token.morphemes):
                        current_token.morphemes[m_idx] = (j2hcj(h2j(current_token.morphemes[m_idx][0])), current_token.morphemes[m_idx][1])
                #print(current_token.FORM, current_token.LEMMA, current_token.morphemes)

            if len(cols) > 3 and (4 not in excludeCols):
                current_token.UPOSTAG = processUnderscore(cols[3])

            if current_token.XPOSTAG == None:   # if SPMRL, we may have already processed XPOSTAG
                if len(cols) > 4 and (5 not in excludeCols):
                    current_token.XPOSTAG = processUnderscore(cols[4])

            if len(cols) > 5 and (6 not in excludeCols):
                if processUnderscore(cols[5]):
                    current_token.FEATS = \
                        [x.strip() for x in cols[5].split('|')]
                else:
                    current_token.FEATS = []
            if len(cols) > 6 and (7 not in excludeCols):
                current_token.HEAD = processUnderscore(cols[6])
                if current_token.HEAD != None:
                    if '-' in current_token.HEAD or '.' in current_token.HEAD:
                        self.logger.warning('line %d: not implemented: HEAD=%s,'
                            ' invalidating sentence' % (ln_num, \
                            current_token.HEAD))

                        invalid_sentence = True
                        continue
                    else:
                        # it's important for parsing that HEAD start at -1
                        current_token.HEAD = int(current_token.HEAD)-1
            if len(cols) > 7 and (8 not in excludeCols):
                current_token.DEPREL = processUnderscore(cols[7])
            if len(cols) > 8 and (9 not in excludeCols):
                # TODO
                current_token.DEPS = processUnderscore(cols[8])
            if len(cols) > 9 and (10 not in excludeCols):
                current_token.MISC = processUnderscore(cols[9])

            current_sentence.tokens.append(current_token)

        # an EOF indicates we need to commit the current sentence
        if current_sentence != None:
            if not invalid_sentence:
                commit(current_sentence)

            current_sentence = None
            current_ID = 0
            invalid_sentence = False

        if self.logStats:
            if self.checkParserConformity:
                self.logger.info('Projectivized %d/%d non-projective sentences' \
                    ' (%.2f%% of set)' % \
                    (well_formed_inst.projectivizedCount, \
                    well_formed_inst.nonProjectiveCount,
                    100.0 * float(well_formed_inst.projectivizedCount) \
                        / float(len(self.sentences))
                    ))

            # if we're even getting rid of malformed sentences in the first
            # place...
            if self.checkParserConformity:
                if not self.keepMalformed:
                    if self.projectivize:
                        # the definition of this variable changes when projectivize
                        # is on
                        self.logger.info('Discarded %d non-well-formed sentences'
                                         % (well_formed_inst.nonWellFormedCount))
                    else:
                        self.logger.info('Discarded %d non-well-formed and ' \
                            ' non-projective sentences' % \
                            (well_formed_inst.nonWellFormedCount))

            self.logger.info('%d valid sentences processed in total' % \
                len(self.sentences))

    '''
    Write the current CoNLL-U data to the specified file descriptor
    '''
    def write(self, fd):
        data = [s.toFileOutput() for s in self.sentences]
        fd.write('\n\n'.join(data))
        fd.flush()

    def __iter__(self):
        index = 0
        while index < len(self.sentences):
            yield self.sentences[index]
            index += 1

class ParsedConllFile(ConllFile):
    def __init__(self, checkParserConformity=False, keepMalformed=False,
                 projectivize=False, logStats=False,
                 enableLemmaMorphemes=False, enableLemmaAsStem=False, compatibleJamo=False,
                 fixMorphemeLabelBugs=False, spmrlFormat=False, koreanAsJamoSeq=False):
        super().__init__(parsed=True,
                         checkParserConformity=checkParserConformity,
                         keepMalformed=keepMalformed,
                         projectivize=projectivize, logStats=logStats,
                         enableLemmaMorphemes=enableLemmaMorphemes,
                         enableLemmaAsStem=enableLemmaAsStem,
                         compatibleJamo=compatibleJamo,
                         fixMorphemeLabelBugs=fixMorphemeLabelBugs,
                         spmrlFormat=spmrlFormat,
                         koreanAsJamoSeq=koreanAsJamoSeq)
