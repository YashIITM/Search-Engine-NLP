import nltk
from nltk.corpus import stopwords
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
import os
import json
# Add your import statements here


class StopwordRemoval():

	def __init__(self):
		nltk.download('stopwords')
		self.stop_words = set(stopwords.words('english'))

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []
		for sentence in text:
			for word in sentence:
				if not word.lower() in self.stop_words:
					stopwordRemovedText.append(word)

		return stopwordRemovedText

segmenter = SentenceSegmentation()
tokenizer = Tokenization()
lemmatizer = InflectionReduction()
stopworder = StopwordRemoval()

CRAN_DOCS_PATH = os.path.relpath("cranfield/cran_docs.json")
docs_json = json.load(open(f"{CRAN_DOCS_PATH}", 'r'))[:]
segmented_bodies = [segmenter.punkt(item["body"]) for item in docs_json]
tokenized_words = [tokenizer.pennTreeBank(body) for body in segmented_bodies]
lemmatized_words = [lemmatizer.reduce(words) for words in tokenized_words]
stopword_removed_words = [stopworder.fromList(words) for words in lemmatized_words]
#print(stopword_removed_words)
