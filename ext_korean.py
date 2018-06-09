"""
Utilities for dealing with Korean
"""

from jamo import h2j, j2hcj

def isJamo(c):
    assert len(c) == 1
    # HANGUL JAMO: (U+1100-U+11FF)
    # HANGUL COMPATIBILITY JAMO: (U+3130-U+318F)
    if ord(c) >= 0x1100 and ord(c) <= 0x11FF:
        return True
    elif ord(c) >= 0x3130 and ord(c) <= 0x318F:
        return True
    else:
        return False

def strIsHangul(s):
    for c in s:
        if not isHangul(c):
            return False
    return True

def strIsHangulOrJamo(s):
    for c in s:
        if not isHangul(c) and not isJamo(c):
            return False
    return True

def isHangul(c):
    return ord(c) >= 0xAC00 and ord(c) <= 0xD7A3

def jamoTail(c):
    assert len(c) == 1
    return int((ord(c) - 44032) % 28)

def jamoVowel(c):
    assert len(c) == 1
    return 1 + (((ord(c) - 44032 - jamoTail(c)) % 588) // 28)

def jamoLead(c):
    assert len(c) == 1
    return 1 + ((ord(c) - 44032) // 588)

def assembleHangul(lead, vowel, tail):
    return chr(tail + (vowel-1)*28 + (lead-1)*588 + 44032)

def removeFinalJamo(c):
    assert len(c) == 1
    lead, vowel = jamoLead(c), jamoVowel(c)
    return assembleHangul(lead, vowel, 0)

def normalizeToCompatJamo(s):
    out = ''
    for c in s:
        if isJamo(c):
            out += j2hcj(c)
        else:
            out += c
    assert len(s) == len(out)
    return out



'''
('닌', '니ㄴ')
'''


def splitOffFinalJamo(c):
    assert len(c) == 1
    assert isHangul(c)
    # important to check if there even is a tail!
    # otherwise, the following call won't work
    if jamoTail(c) > 0:
        # get the compatibility Jamo
        finalJamo = j2hcj(h2j(c)[-1])
        lead, vowel = jamoLead(c), jamoVowel(c)
        return assembleHangul(lead, vowel, 0) + finalJamo
    else:
        return c  # null final: nothing to split off


'''
('군', '굴ㄴ')
'''


def R_irregularOperation(c):
    assert len(c) == 1
    assert isHangul(c)
    # important to check if there even is a tail!
    # otherwise, the following call won't work
    if jamoTail(c) > 0:
        # get the compatibility Jamo
        finalJamo = j2hcj(h2j(c)[-1])
        lead, vowel = jamoLead(c), jamoVowel(c)
        # assemble with R final (R-irregular)
        return assembleHangul(lead, vowel, 8) + finalJamo
    else:
        return c  # null final: nothing to split off


'''
('래', '렇어')
('랬', '렇었')
'''


def H_irregularOperation(c):
    assert len(c) == 1
    assert isHangul(c)
    # important to check if there even is a tail!
    # otherwise, the following call won't work
    if jamoVowel(c) == 2:
        lead, vowel, tail = jamoLead(c), jamoVowel(c), jamoTail(c)
        # lead null consonant is 12
        # 애 -> 엏+어
        # 앴 -> 엏+었
        return assembleHangul(lead, 5, 27) + assembleHangul(12, 5, tail)
    else:
        return c  # null final: nothing to split off


'''
('라', '르아')
('러', '르어')
('떴', '뜨었')
('빴', '쁘았')
'''


def EU_irregularOperation(c):
    assert len(c) == 1
    assert isHangul(c)
    # important to check if there even is a tail!
    # otherwise, the following call won't work
    # only for 아,어
    if jamoVowel(c) == 1 or jamoVowel(c) == 5:
        # get the compatibility Jamo
        finalJamo = j2hcj(h2j(c)[-1])
        lead, vowel, tail = jamoLead(c), jamoVowel(c), jamoTail(c)
        # lead null consonant is 12
        # 아->으+아, or 았->으+았
        # 어->으+어, or 었->으+었
        return assembleHangul(lead, 19, 0) + assembleHangul(12, vowel, tail)
    else:
        return c  # null final: nothing to split off


'''
하다-forms
('했', '하았') [('했', '하았')]
'''


def AE_irregularOperation_1(c):
    assert len(c) == 1
    assert isHangul(c)
    # important to check if there even is a tail!
    # otherwise, the following call won't work
    # only for 애
    if jamoVowel(c) == 2:
        # get the compatibility Jamo
        finalJamo = j2hcj(h2j(c)[-1])
        lead, tail = jamoLead(c), jamoTail(c)
        # lead null consonant is 12
        # 애->아+아, or 앴->아+았
        return assembleHangul(lead, 1, 0) + assembleHangul(12, 1, tail)
    else:
        return c  # null final: nothing to split off


'''
애=>애+어, or 앴=>애+었
'''
'''
def AE_irregularOperation_2(c):
    assert len(c) == 1
    # important to check if there even is a tail!
    # otherwise, the following call won't work
    # only for 애
    if jamoVowel(c) == 2:
        # get the compatibility Jamo
        finalJamo = j2hcj(h2j(c)[-1])
        lead, tail = jamoLead(c), jamoTail(c)
        # lead null consonant is 12
        # 애->아+아, or 앴->아+았
        return assembleHangul(lead, 1, 0) + assembleHangul(12, 1, tail)
    else:
        return c # null final: nothing to split off
'''

# http://www.koreanwikiproject.com/wiki/Vowel_harmony
HARMONY_YANG = 0
HARMONY_YIN = 1
HARMONY_NEUTRAL = 2


def vowelJarmony(c):
    assert len(c) == 1
    assert isHangul(c)
    vowel = jamoVowel(c)

    # YANG
    # ㅏ (a)	ㅑ (ya)	ㅗ (o)	ㅛ (yo)
    # ㅐ (ae)	ㅘ (wa)	ㅚ (oe)	ㅙ (wae)
    if vowel == 1 \
            or vowel == 3 \
            or vowel == 9 \
            or vowel == 13 \
            or vowel == 2 \
            or vowel == 10 \
            or vowel == 12 \
            or vowel == 11:
        return HARMONY_YANG
    # YIN
    # ㅓ (eo)	ㅕ (yeo)	ㅜ (u)	ㅠ (yu)
    # ㅔ (e)	ㅝ (wo)	ㅟ (wi)	ㅞ (we)
    elif vowel == 5 \
            or vowel == 7 \
            or vowel == 14 \
            or vowel == 18 \
            or vowel == 6 \
            or vowel == 15 \
            or vowel == 17 \
            or vowel == 16:
        return HARMONY_YIN

    # otherwise...
    return HARMONY_NEUTRAL


'''
Depending on harmony of vowel, add 아 or 어

('빼', '빼어')
('뺐', '빼었')
...
'''
'''
def addEoOrAOperation(c):
    assert len(c) == 1
    assert isHangul(c)
    if vowelHarmony(c) == HARMONY_YANG:
        # get the compatibility Jamo
        finalJamo = j2hcj(h2j(c)[-1])
        lead, tail = jamoLead(c), jamoTail(c)
        # lead null consonant is 12
        # 애->아+아, or 앴->아+았
        assert None, 'TODO'
        #return assembleHangul(lead, 1, 0) + assembleHangul(12, 1, tail)
    else:
        return c # null final: nothing to split off
'''

'''
('잤', '자았')
'''


def duplicateVowel(c):
    assert len(c) == 1
    assert isHangul(c)
    lead, vowel, tail = jamoLead(c), jamoVowel(c), jamoTail(c)
    # lead null consonant is 12
    return assembleHangul(lead, vowel, 0) + assembleHangul(12, vowel, tail)


# http://www.sayjack.com/blog/2010/06/23/korean-verb-and-adjective-conjugation/


'''
This is for regular contractions

('쳐', '치어')
('렸', '리었')
'''


def decompressVowel(c):
    assert len(c) == 1
    assert isHangul(c)
    # if jamoTail(c) == 0:
    lead, vowel, tail = jamoLead(c), jamoVowel(c), jamoTail(c)

    # lead null consonant is 12
    if vowel == 7:  # 여
        # 여=>이+어, or 였=>이+었
        # print(assembleHangul(lead, 21, 0) + assembleHangul(12, 5, tail))
        return assembleHangul(lead, 21, 0) + assembleHangul(12, 5, tail)
    elif vowel == 2:  # 애
        # 애=>애+어, or 앴=>애+었
        # return assembleHangul(lead, 2, 0) + assembleHangul(12, 5, tail)

        # new rule:
        # 애=>어+어, or 앴=>어+었
        return assembleHangul(lead, 5, 0) + assembleHangul(12, 5, tail)
    elif vowel == 15:  # 워
        # 워=>우+어, or 웠=>우+었
        return assembleHangul(lead, 14, 0) + assembleHangul(12, 5, tail)
    elif vowel == 10:  # 와
        # 와=>오+아, or 왔=>오+았
        return assembleHangul(lead, 9, 0) + assembleHangul(12, 1, tail)
    elif vowel == 11:  # 왜
        # 왜=>외+어, or 왰=>외+었
        return assembleHangul(lead, 12, 0) + assembleHangul(12, 5, tail)
    elif vowel == 6:  # 에
        # 에->이+어, or 엤=>이+었
        return assembleHangul(lead, 21, 0) + assembleHangul(12, 5, tail)
    else:
        # no action possible
        return c
