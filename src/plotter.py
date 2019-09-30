from numpy import zeros
import numpy as np
from src.loader import *

filtered_embedding_matrix = []
not_found = 0
for word, i in tokenizer_encoder.word_index.items():
    try:
        filtered_embedding_matrix.append(word2vec[word])
    except Exception as ex:
      pass
      not_found += 1
filtered_embedding_matrix = np.array(filtered_embedding_matrix)
print(filtered_embedding_matrix.shape)

# getting isotropy
w, v = np.linalg.eig(np.matmul(filtered_embedding_matrix.T, filtered_embedding_matrix))

isotropy = np.sum(np.exp(np.matmul(filtered_embedding_matrix, v)), axis = 0)
print(isotropy.shape)

isotropy.max(), isotropy.min(), isotropy.min() / isotropy.max()

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
filtered_temp = filtered_embedding_matrix - np.average(filtered_embedding_matrix, axis=0)
filtered_principalComponents = pca.fit_transform(filtered_temp)
filtered_principalAxes = pca.components_
filtered_toSubstract = np.matmul(np.matmul(filtered_embedding_matrix, filtered_principalAxes.T), filtered_principalAxes)
filtered_processed = filtered_embedding_matrix - filtered_toSubstract

filtered_w, filtered_v = np.linalg.eig(np.matmul(filtered_processed.T, filtered_processed))
filtered_isotropy = np.sum(np.exp(np.matmul(filtered_processed, filtered_v)), axis = 0)
print(filtered_isotropy.shape)

filtered_isotropy.max(), filtered_isotropy.min(), filtered_isotropy.min() / filtered_isotropy.max()

import seaborn as sns
sns.distplot(np.reshape(filtered_isotropy, (-1)))

import seaborn as sns
sns.distplot(np.reshape(isotropy, (-1)))

word_counts= {k : 0 for k in tokenizer_encoder.index_word.keys()}
word_counts[0] = 0

for sent in x_train:
  for word in sent:
    word_counts[word] += 1

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
filtered_temp = filtered_embedding_matrix - np.average(filtered_embedding_matrix, axis=0)
filtered_principalComponents = pca.fit_transform(filtered_temp)
filtered_principalAxes = pca.components_
filtered_toSubstract = np.matmul(np.matmul(filtered_embedding_matrix, filtered_principalAxes.T), filtered_principalAxes)
filtered_processed = filtered_embedding_matrix - filtered_toSubstract

filtered_principalComponents.shape

import matplotlib.pyplot as plt
plt.scatter(filtered_principalComponents[:, 0], filtered_principalComponents[:, 1], c=[min(i, 200) for i in temp], cmap='bwr', s=1)
plt.colorbar()
plt.title("GLOVE")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig("GloveFreqPlot2")

a = np.arraytokenizer_encoder.word_index.items()

temp = []
for w, i in tokenizer_encoder.word_index.items():
  try:
    k = word2vec[w]
    temp.append(word_counts[i])
  except Exception as ex:
#     print(ex)
    pass

print(len(temp))

len(tokenizer_encoder.word_index.items())



temp

sns.distplot(np.clip(np.array(temp), 0, 100))

!ls

# !mv GloveFrequencyPlot.png gdrive/My\ Drive/hatespeech/

a = np.random.rand(4, 5)
print(a)
print(a/np.linalg.norm(a, ord=2, axis=0, keepdims=True))

c_unnorm = np.random.rand(embedding_dim, 1000)
c_norm = c_unnorm/np.linalg.norm(c_unnorm, ord=2, axis=0, keepdims=True)
z = np.sum(np.exp(np.matmul(filtered_embedding_matrix, c_norm)), axis=0)

z.shape



z2 = np.sum(np.exp(np.matmul(processed, c_norm)), axis=0)
sns.distplot(z)
sns.distplot(z2)

np.exp(np.matmul(processed, c_norm))

