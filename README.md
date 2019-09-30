
# All-but-the-top

Implementation of the paper All-but-the-top from ICLR 2018.

## Instructions to use

To run, use the file runner.py

Libraries used: Keras with tensorflow backend.

### Sample plot

![alt text](https://raw.githubusercontent.com/s1998/All-but-the-top/master/images/gloveFreqPlot.png)

Two principal components of the embeddings with color map for frequency.


## Results 

Results obtained on Davidson et al (2017) using two different embeddings.



| Model   |  	Preprocessed              |    Post-processed             | 
|---------|-------------------------------|-------------------------------|
|         |   P   |   R   |   F1  |  Acc  |   P   |   R   |  F1   |  Acc  |
| AvgPool | 0.871 | 0.892 | 0.874 | 0.882 | 0.855 | 0.887 | 0.862 | 0.887 |
| MaxPool | 0.891 | 0.887 | 0.859 | 0.887 | 0.888 | 0.903 | 0.884 | 0.903 |
| CNN     | 0.885 | 0.903 | 0.880 | 0.903 | 0.890 | 0.905 | 0.892 | 0.905 |
| GRU     | 0.894 | 0.907 | 0.898 | 0.907 | 0.899 | 0.914 | 0.902 | 0.914 |

Effects of using post processing on Glove Embeddings on Davidson et Al(2017)


| Model   |  	Preprocessed              |    Post-processed             | 
|---------|-------------------------------|-------------------------------|
|         |   P   |   R   |   F1  |  Acc  |   P   |   R   |  F1   |  Acc  |
| AvgPool | 0.849 | 0.899 | 0.873 | 0.899 | 0.898 | 0.893 | 0.868 | 0.883 |
| MaxPool | 0.829 | 0.881 | 0.853 | 0.881 | 0.891 | 0.887 | 0.872 | 0.887 |
| CNN     | 0.838 | 0.887 | 0.861 | 0.887 | 0.875 | 0.893 | 0.875 | 0.891 |
| GRU     | 0.854 | 0.904 | 0.878 | 0.904 | 0.910 | 0.903 | 0.881 | 0.903 |

Effects of using post processing on Word2Vec Embeddings on Davidson et Al(2017)
