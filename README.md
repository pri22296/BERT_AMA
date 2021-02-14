# BERT_AMA


Question answering system based on **BERT**. For prediction, the question is searched via the google search API and then the top search results are fed to the model in order to extract candidate answers. Finally scores for the same candidates are combined via the formula mentioned below.

```
TotalScore(text) = SUM(Score(i)/i^2);   1 <= i <= N
```

This ensures that answers from multiple sources are weighted more but still requiring that they be of high quality.

The score for a single answer is given by:

```
Score(span(i,j)) = Max(Pr(start=i) * Pr(end=j)); i<=j
```

Only those answers are selected where:

```
Score(answer) > Score(NO_ANSWER)
```


Further enhancements could include:

1. Currently answers are grouped by exact matched. This could be improved by a clustering based on some kind of similarity metric such as N-gram based similarity, Word vectors based similarity, etc.
2. Another paramter **Tau** could be added such that: `Score(answer) > Score(NO_ANSWER) + Tau`, This was done in the original **BERT** paper. **Tau** could be optimized based on the development set.
3. distillBERT was used for reducing training time but more heavy language models could be used to improve performance.


## Examples

Following are some examples to see BERT_AMA in action.


```python
>>> bert_ama('Who is the founder of google?', source='google', max_urls=5, top_n=1)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>larry page</td>
      <td>0.638208</td>
      <td>[— about a month after donald j. trump was ele...</td>
    </tr>
  </tbody>
</table>
</div>




```python
>>> bert_ama('What is the chemical formula of benzene?', source='google', max_urls=5, top_n=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>c6h6</td>
      <td>1.329239</td>
      <td>[is as shown in the figure below. the chemical...</td>
    </tr>
  </tbody>
</table>
</div>




```python
>>> bert_ama('What is the capital of India?', source='google', max_urls=5, top_n=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>new delhi</td>
      <td>1.207674</td>
      <td>[; malvika singh ; rudrangshu mukherjee ( 2009...</td>
    </tr>
  </tbody>
</table>
</div>




```python
>>> bert_ama('How many carbon atoms does buckminsterfullerene have?', source='google', max_urls=5, top_n=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>sixty</td>
      <td>0.801346</td>
      <td>[is slippery and has a low melting point. buck...</td>
    </tr>
  </tbody>
</table>
</div>




```python
>>> bert_ama('when is the independence day celebrated in india?', source='google', max_urls=5, top_n=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>15th of august</td>
      <td>1.086717</td>
      <td>[independence day of india, which is celebrate...</td>
    </tr>
  </tbody>
</table>
</div>




```python
>>> bert_ama('what is the answer to the ultimate question of life, the universe, and everything?', source='google', max_urls=5, top_n=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>score</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>1.347246</td>
      <td>[slang dictionary 42 [ fawr - tee too ] what d...</td>
    </tr>
  </tbody>
</table>
</div>
