from sklearn.decomposition import PCA
import copy

# Fix as pointed out from here
# https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390#gistcomment-3084023

def get_processed_embeddings(embedding_matrix_orig, n_components = 1):
	pca = PCA(n_components=n_components)
	embedding_matrix = copy.deepcopy(embedding_matrix_orig)
	mean = np.average(embedding_matrix, axis=0)
	temp = embedding_matrix - mean
	principalComponents = pca.fit_transform(temp)
	principalAxes = pca.components_
	toSubstract = np.matmul(np.matmul(embedding_matrix, principalAxes.T), principalAxes)
	processed = embedding_matrix - mean - toSubstract
	return processed
