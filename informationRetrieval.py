from util import *

# Add your import statements here
import math
import numpy as np

class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.freq = None

    def buildIndex(self, docs, docIDs):
        """
			Builds the document index in terms of the document
			IDs and stores it in the 'index' class variable

			Parameters
			----------
			arg1 : list
				A list of lists of lists where each sub-list is
				a document and each sub-sub-list is a sentence of the document
			arg2 : list
				A list of integers denoting IDs of the documents
			Returns
			-------
			None
		"""
        index = {}
        freq = []

        # for i in range(len(docs)):
        #     doc_word_freq = {}
        #     for sentence in docs[i]:
        #         for word in sentence:
        #             if word not in index:
        #                 index[word] = set()
        #             index[word].add(docIDs[i])

        #             if word not in doc_word_freq:
        #                 doc_word_freq[word] = 1
        #             else:
        #                 doc_word_freq[word] += 1
                    

        #     freq.append(doc_word_freq)

        for i in range(len(docs)):
            doc_word_freq = {}
            for sentence in docs[i]:
                if sentence not in index:
                    index[sentence] = set()
                    index[sentence].add(docIDs[i])

                if  sentence not in doc_word_freq:
                    doc_word_freq[sentence] = 1
                else:
                    doc_word_freq[sentence] += 1
                    

            freq.append(doc_word_freq)

        self.index = index
        #Freq is the number of times a word occurs in a particular document. SO its a list of dictionaries
        #and the key value pairs of the dictionaries are the word and the number of times they occur 
        #for that document.
        self.freq = freq
        self.wordset = list(self.index.keys())
        self.docs = docs

    # def tf_idf(self, docs, query):
    #     #This function generates the TF-IDF metric for query
    #     docs_len = len(docs)
    #     #Number of times a word occurs in all the documents combined
    #     relv_len = len(self.index[query]) + 1
    #     idf = math.log(docs_len / relv_len, 10)

	# 	#no_of_words_in_doc = []
    #     tfs = []
    #     for i in range(len(docs)):
    #         total_words = 0
    #         for sentence in docs[i]:
    #             total_words += len(sentence)
	# 		# no_of_words_in_doc.append(total_words)
    #         doc_query_freq = self.freq[i].get(query, 0)
    #         tf = doc_query_freq / total_words
    #         tfs.append(tf)
    #     # print(tfs,'1')
        
    #     tf_idfs = []
    #     for tf in tfs:
    #         tf_idfs.append(tf * idf)

    #     return tf_idfs
    
    def vector_model(self,docs,query):
        """
        Creates vector model of various documents in docs and also a query to aid in helping to the weights
        used to rank the documents
        """
        term_freq_docs=[]
        def computeTF(doc):
            raw_tf = dict.fromkeys(self.wordset,0)
            norm_tf = {}
            bow = len(doc)
            # print(doc,"#")
            for word in doc:
                raw_tf[word]+=1   ##### term frequency
            for word, count in raw_tf.items():
                norm_tf[word] = count / (float(bow) +1) ###### Normalized term frequency
            return raw_tf, norm_tf
        def computeTF_query(query):
            raw_tf = dict.fromkeys(self.wordset,0)
            norm_tf = {}
            bow = len(query)
            for word in query:
                if word in raw_tf:
                    raw_tf[word]+=1   ##### term frequency
                else:
                    continue
            for word, count in raw_tf.items():
                norm_tf[word] = count / (float(bow) +1) ###### Normalized term frequency
            return raw_tf, norm_tf 
        for doc in docs:
            # final_freq_doc = dict.fromkeys(self.wordset,0) 
                
            norm_tf = computeTF(doc)[0]# 1 = normalised term frequency, 0 = raw term frequecy
            # final_freq_doc = {x: final_freq_doc.get(x, 0) + norm_tf.get(x, 0)
            #         for x in set(final_freq_doc).union(norm_tf)}
            # format_dict = dict.fromkeys(self.wordset,0)
            # for key in format_dict:
            #     format_dict[key] = final_freq_doc[key]
            term_freq_docs.append(norm_tf)


        term_freq_query=[]
        # print(query,"#")
        norm_tf = computeTF_query(query)[0]# 1 = normalised term frequency, 0 = raw term frequecy
        term_freq_query.append(norm_tf)
        #print(term_freq_docs)
        # The term_freq_docs is a list of dictionaries containing the words in wordset as keys and
        # their normalised term frequencies as their corresponding value pair. 
        self.term_freq_docs = term_freq_docs

        # The term_freq_query is a list of dictionaries containing the words in wordset as keys and 
        # their raw term frequencies as their corresponding value pair
        self.term_freq_query = term_freq_query[0]

        def computeIdf(doc_list):
            """
            Here doc list is the list of term frequency vector of docs
            Returns the IDF vector for the particular query that was used to generate the vector
            model to start with
            """
            idf={}
            idf = dict.fromkeys(doc_list[0].keys(),float(0))
    
            for doc in doc_list:
                for word, val in doc.items():
                    if val > 0:
                        idf[word] += 1
                
            for word, val in idf.items():
                idf[word] = math.log10(len(doc_list) / float(val))
            return idf
        self.idf = computeIdf(self.term_freq_docs)


    def rank(self, queries):
        """
				Rank the documents according to relevance for each query

				Parameters
				----------
				arg1 : list
					A list of lists of lists where each sub-list is a query and
					each sub-sub-list is a sentence of the query


				Returns
				-------
				list
					A list of lists of integers where the ith sub-list is a list of IDs
					of documents in their predicted order of relevance to the ith query
		"""

        doc_IDs_ordered = []

        # Fill in code here
        def cosine_similarity(dict_1,dict_2):
            """
            This function returns the similarity value of 2 dictionaries that contain the value as 
            weights
            """
            vec_1 = list(dict_1.values())
            vec_2 = list(dict_2.values())
            numerator = np.dot(vec_1,vec_2)
            denominator = np.linalg.norm(np.array(vec_1))*np.linalg.norm(np.array(vec_2))
            return numerator/denominator



        weights = []
        
        for query in queries:
            # creating the vector model of all documents in docs for a particular query
            self.vector_model(self.docs,query)
            #Lets create a weight vector for each query
            query_weight = dict.fromkeys(self.wordset,0) 
            for key in query_weight:
                query_weight[key] = self.term_freq_query[key]*self.idf[key]

            #Lets create matrix of weight vectors for all documents
            docs_weight = []
            for i in range(len(docs)):
                doc_weight = dict.fromkeys(self.wordset,0)
                for key in doc_weight:
                    doc_weight[key] = self.term_freq_docs[i][key]*self.idf[key]
                docs_weight.append(doc_weight)
            
            #query_weight is dictionary of the weight values for the query
            #docs_weight is a list of dictionary of weight values 
            similarity_docs = []
            for i in range(len(docs)):
                similarity = cosine_similarity(query_weight,docs_weight[i])
                similarity_docs.append(similarity)
                #print(similarity,i)
            
            # Now we need to sort the array "similarity_docs" and return corresponding index matrix
            rank_list = np.argsort(np.array(similarity_docs))
            doc_IDs_ordered.append(list(rank_list[::-1]))

        return doc_IDs_ordered


doc1 = ["herbivores are typically plant eaters and not meat eaters"," plant eaters are not meat eaters"]
doc2 = ["carnivores are typically meat eaters and not plant eaters","deers are not meat eaters"]
doc3 = ["deers eat grass and leaves","plants include grass and leaves"]

#we already have the tokens of the docs
tokens1 = ["herbivores","typically", "plant", "eaters", "meat", "eaters"]
tokens2 = ["carnivores", "typically", "meat", "eaters", "plant", "eaters"]
tokens3 = ["deers", "eat", "grass", "leaves"]
docs = [tokens2,tokens1,tokens3]


#wordset = set(tokens1).union(set(tokens2)).union(set(tokens3))
s = [[["herbivores"], ["are"], ['are'], ["typically"], ["plant"], ["eaters"], ["and"], ["not"], ["meat"], ["eaters"]],
     [["carnivores"], ["are"], ['are'], ["typically"], ["meat"], ["eaters"], ["and"], ["not"], ["plant"], ["eaters"]],
     [["deers"], ["eat"], ["grass"], ["and"], ["leaves"]]
     ]

# s_ids = [1, 2, 3]
# ir = InformationRetrieval()

# ir.buildIndex(docs, s_ids)
# # # # print(ir.freq)
# # # print(docs[0])
# # # #print(ir.freq[1])
# ir.vector_model(docs, ["plant","eaters"])
# print(ir.rank([["plant","eaters"]]))
# # # print(ir.term_freq_query)