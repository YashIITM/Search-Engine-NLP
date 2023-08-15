from util import *
import math
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		true_positives = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] == true_doc_IDs[i]:
				true_positives += 1

		precision = true_positives / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		for query_id in query_ids:
			
			ground_truth = []
			for query_dict in qrels:
				if query_dict["query_num"] == query_id:
					ground_truth.append([query_dict["position"], query_dict["id"]])


			ground_truth.sort()

			true_positives = 0
			precision_k = 0
			for i in range(k):
				if doc_IDs_ordered[i] == ground_truth[i][1]:
					true_positives += 1
					curr_precision = true_positives / (i+1)
					precision_k += curr_precision
			
			meanPrecision = precision_k / true_positives
		
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		
		true_positives = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] == true_doc_IDs[i]:
				true_positives += 1

		recall = true_positives / len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		for query_id in query_ids:
			
			ground_truth = []
			for query_dict in qrels:
				if query_dict["query_num"] == query_id:
					ground_truth.append([query_dict["position"], query_dict["id"]])


			ground_truth.sort()

			true_positives = 0
			recall_k = 0
			for i in range(k):
				if doc_IDs_ordered[i] == ground_truth[i][1]:
					true_positives += 1
					curr_recall = true_positives / len(ground_truth)
					recall_k += curr_recall
			
			meanRecall = recall_k / true_positives
		
		return meanRecall

	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		precison = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = 2*precison*recall/(precison+recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		precison = self.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
		recall = self.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
		meanFscore = 2*precison*recall/(precison+recall)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		DCG=0
		for i in range(len(query_doc_IDs_ordered)):
			DCG+=query_doc_IDs_ordered[i] / math.log(i+2, 2)

		IDCG = 0
		for i in range(len(true_doc_IDs)):
			IDCG+=true_doc_IDs[i] / math.log(i+2, 2)
		nDCG = DCG/IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
		
		meanNDCG = 0
		for query_id in query_ids:

			ground_truth = []
			for query_dict in qrels:
				if query_dict["query_num"] == query_id:
					ground_truth.append([query_dict["position"], query_dict["id"]])


			ground_truth.sort()

			DCG=0
			for i in range(len(doc_IDs_ordered)):

				DCG += doc_IDs_ordered[query_id][i] / math.log(i+2, 2)

			IDCG=0
			for i in range(len(doc_IDs_ordered)):

				IDCG += ground_truth[i][0] / math.log(ground_truth[i][1]+1, 2)
			meanNDCG += (DCG/IDCG)


		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		true_positives = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] == true_doc_IDs[i]:
				true_positives += 1

		avgPrecision = true_positives / k

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""
		precision_sum = 0
		for query in query_ids:
			mean_precision = self.meanPrecision(doc_IDs_ordered, query_ids, q_rels, k)	
			precision_sum += mean_precision

		meanAveragePrecision = precision_sum/len(query_ids)


		return meanAveragePrecision

