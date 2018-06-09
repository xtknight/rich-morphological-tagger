'''
Simple varlen oracle

to convert L1 to L2
(FORM to LEMMA)
'''

def varlen_oracle(L1, L2):
	O = []
	for i in range(max(len(L1), len(L2))):
		if len(L1) > i:
			if len(L2) > i:
				if L1[i] == L2[i]:
					O.append('KEEP')
				else:
					O.append('MOD:' + L2[i])
			else:
				# ignore this part of L1
				O.append('NOOP')
		else:
			# now we have a problem. append all mods to last element of L1??
			# or leave a buffer??
			if len(L2) > i:
				# this almost NEVER happens unless somehow the LEMMA is longer
				# than the FORM
				# we can leave some buffer in case??
				O.append('MOD:' + L2[i]) # means ADD:
			else:
				# this shouldn't happen
				assert None, 'i not in L1 or L2???'

	return O

'''
Restore LEMMA from given FORM and actions list
'''
def restore_from_actions(L1, A, rejoin=True):
	L2 = []
	for idx, a in enumerate(A):
		if a == 'KEEP':
			assert len(L1) > idx
			L2.append(L1[idx])
		elif a == 'NOOP':
			assert len(L1) > idx
		elif a.startswith('MOD:'):
			L2.append(a[4:])
		else:
			assert None, 'unknown action: ' + str(a)
	if rejoin:
		return ''.join(L2)
	else:
		return L2

'''
Restore LEMMA and also non-stem part from given FORM and actions list
'''
def restore_from_actions_plusnonstem(L1, A):
	L2 = []
	current_stem = []
	current_nonstem = []
	for idx, a in enumerate(A):
		if a == 'KEEP':
			assert len(L1) > idx

			if len(current_nonstem) > 0:
				L2.append((''.join(current_nonstem), 'NONSTEM'))
				current_nonstem = []

			current_stem.append(L1[idx])

			#L2.append(L1[idx])
		elif a == 'NOOP':
			assert len(L1) > idx
			if len(current_stem) > 0:
				L2.append((''.join(current_stem), 'STEM'))
				current_stem = []

			current_nonstem.append(L1[idx])
		elif a.startswith('MOD:'):
			## @@ TODO: combine these if possible as well

			# add mod target to stem
			current_stem.append(a[4:])
			if len(current_stem) > 0:
				L2.append((''.join(current_stem), 'STEM'))
				current_stem = []

			# @@TODO: exception: not necessary to duplicate chars if same
			# we want a clean split for parçacı|lar
			# but maintain consistency for now

			# add original part to non-stem
			current_nonstem.append(L1[idx])

			#L2.append(a[4:])
		else:
			assert None, 'unknown action: ' + str(a)

	# both shouldn't be...
	assert len(current_stem) == 0 or len(current_nonstem) == 0

	if len(current_stem) > 0:
		L2.append((''.join(current_stem), 'STEM'))
	elif len(current_nonstem) > 0:
		L2.append((''.join(current_nonstem), 'NONSTEM'))

	return L2

'''
Return LEMMA + changed or unused other segments

Resegment based on what was removed and kept
'''
'''
def split_at_stem_from_actions(L1, A, rejoin=True):
	#contig_stem = restore_from_actions(L1, A)
	# now split only based on KEEP portions
'''

'''
Split elements at even ngram intervals
'''
def split_elems(L, ngram=1):
	assert type(L) is list
	S = []
	for i in range(0, len(L), ngram):
		S.append(L[i:i+ngram])
	return S

'''
Split L at specified indices
'''
def split_at(L, indices, joinSubLists=True):
	assert type(L) is list
	S = []
	i = 0
	lastindex = 0
	while i < len(indices):
		#if lastindex > 0:
		# disallow first entry being 0
		assert indices[i] > lastindex

		subL = L[lastindex:indices[i]]
		#print('subL', subL)
		if len(subL) > 0:
			if joinSubLists:
				S.append(''.join(subL))
			else:
				S.append(subL)
		lastindex = indices[i]
		i += 1

	# add last piece
	subL = L[lastindex:]
	if len(subL) > 0:
		if joinSubLists:
			S.append(''.join(subL))
		else:
			S.append(subL)

	return S

#varlen_oracle(list('hello')[::3]

#print(split_elems(list('hello'), 2))
# parçacıklar -> parçacık
#FORM_split = split_at(list('parçacıklar'), [3,8])
#LEMMA_split = split_at(list('parçacık'), [3,8])

'''
split_pieces = [
	'aaaaa',
	'bbb',
	'bbc',
	'bc',
	'c',
	'd',
]
# for consistency:
# sort by alphabetical order
split_pieces = sorted(split_pieces)
# then sort by longest first (stable sort, so alphabetical order should be maintained)
split_pieces = sorted(split_pieces, key=len, reverse=True)

'''


'''
Get the index of the longest splittable 'thing' in m
'''
'''def get_longest_split_loc(m):
	for loc in split_pieces: # ASSUMES SORTED BY LENGTH
		#if len(loc) > 1: # doing by one char is risky
		if loc in m:
			return m.index(loc)
	return None
'''

'''
Get all indexes that we can split m into
'''
def get_all_split_pieces(m):
	S = set()
	for loc in split_pieces:
		if loc in m:
			ind = m.index(loc)
			if ind > 0: # we don't split at 'beginning'
				S.add(ind)
	return sorted(S)


'''
Get all indexes that we can split m into
'''
def get_longest_split_pieces(m, split_pieces):
	assert type(m) is list
	# keep track of invalidated pieces
	# when we find a long piece, we then mark the locations it's responsible for
	# as invalid
	# that way, a shorter piece can't also be responsible for it
	# a simpler way would probably be to replace the longest subsequences by
	# invalid chars repeatedly, but this is cleaner and easily traceable
	finished = [False for _ in m]
	S = set()
	for piece in split_pieces:
		assert len(piece) > 0
		if piece in m:
			#print(piece, 'is in word')
			ind = m.index(piece)
			## check if this is an invalidated location
			invalid = False
			for k in range(len(piece)):
				if finished[ind+k]:
					print('Found subsequence %s but was invalidated by prior/longer subsequence' % piece)
					invalid = True
					break

			if not invalid:		
				for k in range(len(piece)):
					finished[ind+k] = True

				if ind > 0: # we don't split at 'beginning' (index 0 is considered a given)
					S.add(ind)
	# return the split indices in order
	return sorted(S)


'''
import sys
split_indices = get_longest_split_pieces('aaaaabbbccd', split_pieces)
print(split_indices)
print(split_at(list('aaaaabbbccd'), split_indices))
'''

'''
Get locations to split (for FORM based on prefixes and suffixes)

Recursive
'''
'''
def get_split_locations(FORM):
	S = []

	split_index = get_longest_split_loc(FORM)
	if split_index != None:
		S.append(split_index)
		# also recurse and see if we can split further
		piece1, piece2 = split_at(FORM, split_index)
		

	return S
'''

'''
FORM_split = split_at(list('parçacıklar'), [2,4,6,8,10,12,14])
LEMMA_split = split_at(list('parçacız'), [2,4,6,8,10,12,14])

print('FORM_split', FORM_split)
print('LEMMA_split', LEMMA_split)
print('FORM->LEMMA', varlen_oracle(FORM_split, LEMMA_split))
print('LEMMA->FORM', varlen_oracle(LEMMA_split, FORM_split))
#print(varlen_oracle(LEMMA_split, LEMMA_split))
#print(varlen_oracle(FORM_split, FORM_split))
'''

'''

FORM_split = split_at(list('parçacıklar'), [2,4,6,8,10,12,14])
LEMMA_split = split_at(list('parçacız'), [2,4,6,8,10,12,14])

print('FORM_split', FORM_split)
print('LEMMA_split', LEMMA_split)
print('FORM->LEMMA', varlen_oracle(FORM_split, LEMMA_split))
print('LEMMA->FORM', varlen_oracle(LEMMA_split, FORM_split))

assert 'parçacız' == restore_from_actions(FORM_split, varlen_oracle(FORM_split, LEMMA_split))
assert 'parçacıklar' == restore_from_actions(LEMMA_split, varlen_oracle(LEMMA_split, FORM_split))

#print(varlen_oracle(LEMMA_split, LEMMA_split))
#print(varlen_oracle(FORM_split, FORM_split))
'''

FORM_split = split_at(list('parçacılar'), [2,4,6,8,10,12,14])
LEMMA_split = split_at(list('zzrçacı'), [2,4,6,8,10,12,14])

print('FORM_split', FORM_split)
o = varlen_oracle(FORM_split, LEMMA_split)
print('FORM->LEMMA', o)

print(restore_from_actions_plusnonstem(FORM_split, o))
