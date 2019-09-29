from annoy import AnnoyIndex
from gensim.matutils import softcossim
from sklearn.metrics.pairwise import cosine_similarity
from queue import Queue
import numpy as np


class QueryModel(object):
    dataset_size = 0
    nodes = []
    similarity_matrix = []
    # similarity_matrix.re
    adj = []
    graph_edge_threshold = 0.3

    def __init__(self, vector_dimentions):
        self.vector_dimentions = vector_dimentions
        self.all_vectors = AnnoyIndex(vector_dimentions, 'angular')
        self.ANN_datastruct = self.all_vectors
        self.ANN_datastruct.build(10)
        self.ANN_datastruct.save('docs.ann')

    def nodesComparer(self, x, y):
        return cosine_similarity([x], [y])[0][0]

    def query_nodeComparer(self, x, y):
        return cosine_similarity([x], [y])[0][0]

    def change_graph_edge_threshold(self, graph_edge_threshold):
        '''
        Change the threshold similarity between datasets so that they must be connected in the graph
        '''
        # print(self.adj)
        if graph_edge_threshold != self.graph_edge_threshold:
            self.graph_edge_threshold = graph_edge_threshold
            for i in range(self.dataset_size):
                self.adj[i].clear()
            for i in range(self.dataset_size):
                for j in range(i):
                    if self.similarity_matrix[i][j] >= graph_edge_threshold:
                        self.adj[i].append(j)
                        self.adj[j].append(i)
        # print(self.adj)

    def addToDataset(self, list_of_inferred_vectors):
        '''
        array of tuples -> [(id, inferred_vectors)]
        '''
        # print("LOL")
        all_vectors = AnnoyIndex(self.vector_dimentions, 'angular')
        final = self.dataset_size+len(list_of_inferred_vectors)
        for i in range(0,self.dataset_size):
            # print("LOL")
            self.similarity_matrix[i] = np.resize(self.similarity_matrix[i], final)
            # np.concatenate((self.similarity_matrix[i],np.zeros(len(list_of_inferred_vectors)))
            all_vectors.add_item(i, self.nodes[i][1])
        # self.similarity_matrix.append([])
        # print(self.dataset_size)
        # print(self.similarity_matrix)
        for newData in list_of_inferred_vectors:
            all_vectors.add_item(self.dataset_size, newData[1])
            self.similarity_matrix.append(np.zeros(final))
            self.nodes.append(newData)
            self.adj.append([])
            for prev_node in range(self.dataset_size+1):
                similarity = self.nodesComparer(
                    self.nodes[prev_node][1], newData[1])
                # print(prev_node)
                # print(self.dataset_size)
                self.similarity_matrix[prev_node][self.dataset_size] = similarity
                self.similarity_matrix[self.dataset_size][prev_node] = similarity
                if similarity >= self.graph_edge_threshold and prev_node != self.dataset_size:
                    self.adj[prev_node].append(self.dataset_size)
                    self.adj[self.dataset_size].append(prev_node)
            self.dataset_size += 1
        self.ANN_datastruct = all_vectors
        self.ANN_datastruct.build(10)
        self.ANN_datastruct.save('docs.ann')
        # print(self.similarity_matrix)

    def query(self, query_inferred_vector, query_similarity_threshold=0.3):
        '''
        starts by finding 10 nearest neighbour throung ANN, 
        then does BFS from these until similarity is above query_similarity_threshold
        '''
        self.ANN_datastruct = AnnoyIndex(self.vector_dimentions, 'angular')
        self.ANN_datastruct.load('docs.ann')
        approx_nearest_neighbour = self.ANN_datastruct.get_nns_by_vector(query_inferred_vector, 10)
        q = Queue()
        # print("*********")
        # print(approx_nearest_neighbour)
        for i in approx_nearest_neighbour:
            q.put(i)
        answer = []
        checked = np.zeros(self.dataset_size)
        while not q.empty():
            i = q.get()
            if checked[i]==0:
                checked[i]=1
                similarity = self.query_nodeComparer(query_inferred_vector, self.nodes[i][1])
                if similarity >= query_similarity_threshold:
                    answer.append((similarity,self.nodes[i][0]))
                    for possible in self.adj[i]:
                        if checked[possible] == 0:
                            q.put(possible)
        answer = sorted(answer, reverse=True)
        return answer
