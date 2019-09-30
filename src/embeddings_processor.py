from sklearn.decomposition import PCA
import copy

def get_processed_embeddings(embedding_matrix_orig, n_components = 1):
	pca = PCA(n_components=n_components)
	embedding_matrix = copy.deepcopy(embedding_matrix_orig)
	temp = embedding_matrix - np.average(embedding_matrix, axis=0)
	principalComponents = pca.fit_transform(temp)
	principalAxes = pca.components_
	toSubstract = np.matmul(np.matmul(embedding_matrix, principalAxes.T), principalAxes)
	processed = embedding_matrix - toSubstract
	return processed
