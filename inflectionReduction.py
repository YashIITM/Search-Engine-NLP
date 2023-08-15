import nltk
from nltk.stem import WordNetLemmatizer
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
import os
import json
# Add your import statements here


class InflectionReduction:

    def __init__(self):
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

    def reduce(self, text):
        """
        Stemming/Lemmatization
        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """

        reducedText = [[self.lemmatizer.lemmatize(word).replace('/', '').replace('-', '') for word in sentence]
                       for sentence in text]

        return reducedText

segmenter = SentenceSegmentation()
tokenizer = Tokenization()
lemmatizer = InflectionReduction()

CRAN_DOCS_PATH = os.path.relpath("cranfield/cran_docs.json")
docs_json = json.load(open(f"{CRAN_DOCS_PATH}", 'r'))[:]
segmented_bodies = [segmenter.punkt(item["body"]) for item in docs_json]
tokenized_words = [tokenizer.pennTreeBank(body) for body in segmented_bodies]
lemmatized_words = [lemmatizer.reduce(words) for words in tokenized_words]
