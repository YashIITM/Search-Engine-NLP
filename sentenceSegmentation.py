from nltk.tokenize import punkt
import json
import os
# Add your import statements here


class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        text(arg1) : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """

        segmentedText = [a.strip(' ') for a in text.replace('? ', '? |').replace('. ', '. |').split('|')]
        if '' in segmentedText:
            segmentedText.remove('')

        return segmentedText

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        text(arg1) : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """

        sent_splitter = punkt.PunktSentenceTokenizer()
        segmentedText = sent_splitter.tokenize(text)

        return segmentedText


segmenter = SentenceSegmentation()

CRAN_DOCS_PATH = os.path.relpath("cranfield/cran_docs.json")
docs_json = json.load(open(f"{CRAN_DOCS_PATH}", 'r'))[:]
bodies = [item["body"] for item in docs_json]
mismatch_count = 0

for body in bodies:
    naive_res = segmenter.naive(body)
    punkt_res = segmenter.punkt(body)
    if naive_res != punkt_res:
        mismatch_count += 1

    # Printing some mismatch examples
    '''
    for i in range(min(len(naive_res), len(punkt_res))):
        if naive_res[i] != punkt_res[i]:
            if len(naive_res[i]) < len(punkt_res[i]):
                print(bodies.index(body), naive_res[i])
                print(punkt_res[i])
            break
    '''

print(f"Naive top-down approach gives wrong segmentation in {mismatch_count} sentences out of {len(bodies)}")
print(f"Sentence Segmentation Accuracy: {(len(bodies) - mismatch_count) / len(bodies) * 100}%")
