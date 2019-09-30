
# All-but-the-top

Implementation of the paper [All-but-the-top](https://openreview.net/forum?id=HkuGJ3kCb) from ICLR 2018.

## Instructions to use

To run, use the file runner.py

Libraries used: Keras with tensorflow backend.

### Sample plot

![image-of-principal-components](https://raw.githubusercontent.com/s1998/All-but-the-top/master/images/gloveFreqPlot.png)

Two principal components of the embeddings with color map for frequency.


## Results 

Results obtained on Davidson et al (2017) using two different embeddings.

<table>
  <tr>
  	<td> Model </td>
  	<td colspan="4"> Preprocessed </td>
  	<td colspan="4"> Postprocessed </td>
  </tr>
  <tr>
  	<td></td> 
  	<td>  P  </td>
  	<td>  R  </td>
  	<td>  F1  </td> 
  	<td>  Acc  </td> 
  	<td>  P  </td> 
  	<td>  R  </td> 
  	<td>  F1  </td> 
  	<td>  Acc  </td> 
  </tr>
  <tr>
  	<td>AvgPool </td> 
  	<td> 0.871  </td>
  	<td> 0.892  </td>
  	<td> 0.874  </td> 
  	<td> 0.882  </td> 
  	<td> 0.855  </td> 
  	<td> 0.887  </td> 
  	<td> 0.862  </td> 
  	<td> 0.887  </td> 
  </tr>
  <tr>
  	<td>MaxPool </td> 
  	<td> 0.891  </td>
  	<td> 0.887  </td>
  	<td> 0.859  </td> 
  	<td> 0.887  </td> 
  	<td> 0.888  </td> 
  	<td> 0.903  </td> 
  	<td> 0.884  </td> 
  	<td> 0.903  </td> 
  </tr>
  <tr>
  	<td>CNN     </td> 
  	<td> 0.885  </td>
  	<td> 0.903  </td>
  	<td> 0.880  </td> 
  	<td> 0.903  </td> 
  	<td> 0.890  </td> 
  	<td> 0.905  </td> 
  	<td> 0.892  </td> 
  	<td> 0.905  </td> 
  </tr>
  <tr>
  	<td>GRU     </td> 
  	<td> 0.894  </td>
  	<td> 0.907  </td>
  	<td> 0.898  </td> 
  	<td> 0.907  </td> 
  	<td> 0.899  </td> 
  	<td> 0.914  </td> 
  	<td> 0.902  </td> 
  	<td> 0.914  </td> 
  </tr>
</table>

Effects of using post processing on Glove Embeddings on Davidson et Al(2017)


<table>
  <tr>
  	<td> Model </td>
  	<td colspan="4"> Preprocessed </td>
  	<td colspan="4"> Postprocessed </td>
  </tr>
  <tr>
  	<td></td> 
  	<td>  P  </td>
  	<td>  R  </td>
  	<td>  F1  </td> 
  	<td>  Acc  </td> 
  	<td>  P  </td> 
  	<td>  R  </td> 
  	<td>  F1  </td> 
  	<td>  Acc  </td> 
  </tr>
  <tr>
  	<td> AvgPool </td> 
  	<td> 0.849 </td>
  	<td> 0.899 </td>
  	<td> 0.873 </td>
  	<td> 0.899 </td>
  	<td> 0.898 </td>
  	<td> 0.893 </td>
  	<td> 0.868 </td>
  	<td> 0.883 </td>
  </tr>
  <tr>
  	<td> MaxPool </td> 
  	<td> 0.829 </td>
  	<td> 0.881 </td>
  	<td> 0.853 </td>
  	<td> 0.881 </td>
  	<td> 0.891 </td>
  	<td> 0.887 </td>
	<td> 0.872 </td>
	<td> 0.887 </td>
  </tr>
  <tr>
  	<td> CNN     </td> 
  	<td> 0.838 </td>
  	<td> 0.887 </td>
  	<td> 0.861 </td>
  	<td> 0.887 </td>
  	<td> 0.875 </td>
  	<td> 0.893 </td>
	<td> 0.875 </td>
	<td> 0.891</td>
  </tr>
  <tr>
  	<td> GRU     </td> 
  	<td> 0.854 </td>
  	<td> 0.904 </td>
  	<td> 0.878 </td>
  	<td> 0.904 </td>
  	<td> 0.910 </td>
  	<td> 0.903 </td>
	<td> 0.881 </td>
	<td> 0.903</td>
  </tr>
</table>

Effects of using post processing on Word2Vec Embeddings on Davidson et Al(2017)
