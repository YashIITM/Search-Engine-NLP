import json
from sentenceSegmentation import SentenceSegmentation
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import WhitespaceTokenizer
import os
# Add your import statements here


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		white_space = WhitespaceTokenizer()
		tokenizedText = []
		for sentence in text:
			tokenizedText.append(white_space.tokenize(sentence))

		return tokenizedText

	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		penn = TreebankWordTokenizer()
		tokenizedText = []
		for sentence in text:
			tokenizedText.append(penn.tokenize(sentence))

		return tokenizedText


segmenter = SentenceSegmentation()
tokenizer = Tokenization()

CRAN_DOCS_PATH = os.path.relpath("cranfield/cran_docs.json")
docs_json = json.load(open(f"{CRAN_DOCS_PATH}", 'r'))[:]
segmented_bodies = [segmenter.punkt(item["body"]) for item in docs_json]
mismatch_count = 0

for body in segmented_bodies:
	naive_res = tokenizer.naive(body)
	penn_res = tokenizer.pennTreeBank(body)
	if naive_res != penn_res:
		mismatch_count += 1

		# Printing some mismatch examples
		'''
		if mismatch_count < 5:
			for i in range(len(naive_res)):
				if naive_res[i] != penn_res[i]:
					print(naive_res[i])
					print(penn_res[i])
				print(body[i])
		'''

print(f"Naive top-down approach gives wrong tokenization in {mismatch_count} sentences out of {len(segmented_bodies)}")
print(f"Tokenization Accuracy: {(len(segmented_bodies) - mismatch_count) / len(segmented_bodies) * 100}%")

