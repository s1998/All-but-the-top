import pandas as pd
import numpy as np

data_r = pd.read_csv("./labeled_data.csv")
data = data_r[['class', 'tweet']]
maxlen_seq = 100
embedding_dim = 100

def process_data(data):
    
    import re
    pattern  = re.compile('@[a-zA-Z0-9_]*')
    pattern4 = re.compile('\s+')
    pattern5 = re.compile(r"(.)\1{2,}", re.DOTALL)
    pattern6 = re.compile(r'http\S+')
    pattern8 = re.compile('#(\w+)')

    def process_string(line_):
      line = pattern.sub(r'<user> ', line_)
      line4 = pattern4.sub(' ', line)
      line5 = pattern5.sub(r"\1\1", line4)
      line6 = pattern6.sub(' <url> ', line5).lower()
      line8 = pattern8.sub(r'<hashtag> \1 <hashtag>', line6)
      return line8


    data = data[data['class'].isin([0, 1, 2, 3])]
    data = data.reset_index(drop = True)
    data['tweet'] = data['tweet'].apply(process_string)
    data = data[['class', 'tweet']]
    print(data[:100])
    return data
  
data = process_data(data)

!wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
!unzip glove.twitter.27B.zip
!sed -i '1s/^/1193514 100\n/' glove.twitter.27B.100d.txt
!sed -i '1s/^/1193514 200\n/' glove.twitter.27B.200d.txt
from gensim.models.keyedvectors import Word2VecKeyedVectors
word2vec = Word2VecKeyedVectors.load_word2vec_format("./glove.twitter.27B.100d.txt")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['tweet'], data['class'], 
                                                    test_size=0.10, random_state=76, stratify=data['class'])

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, 
                                                    test_size=0.10, random_state=76, stratify=y_train)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(data['tweet'])

x_train = tokenizer_encoder.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen = maxlen_seq, padding = 'post')

# use same tokenizer defined on train for tokenization of test set
x_test = tokenizer_encoder.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen = maxlen_seq, padding='post')

# use same tokenizer defined on train for tokenization of validation
x_dev = tokenizer_encoder.texts_to_sequences(x_dev)
x_dev = pad_sequences(x_dev, maxlen = maxlen_seq, padding='post')

# y_train = keras.utils.to_categorical(train_data['Label'], num_classes)
# y_test = keras.utils.to_categorical(test_data['Label'], num_classes)

n_words = len(tokenizer_encoder.word_index) + 1

from numpy import zeros
import numpy 

embedding_matrix = zeros((n_words, embedding_dim))
not_found = 0
for word, i in tokenizer_encoder.word_index.items():
    try:
        embedding_matrix[i] = word2vec[word]
    except Exception as ex:
        embedding_matrix[i] = numpy.random.normal(embedding_dim)
        not_found += 1

print("Not found vectors for {} out of {} words".format(not_found, n_words))
