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


## Example

```
>>> bert_ama('Who is the founder of google?', source='google', max_urls=5, top_n=1)
    text 	score 	context
2 	larry page 	0.638208 	[— about a month after donald j. trump was ele...

>>> bert_ama('What is the chemical formula of benzene?', source='google', max_urls=5, top_n=1)
 	text 	score 	context
1 	c6h6 	1.329239 	[is as shown in the figure below. the chemical...

>>> bert_ama('What is the capital of India?', source='google', max_urls=5, top_n=1)
 	text 	score 	context
1 	new delhi 	1.207674 	[; malvika singh ; rudrangshu mukherjee ( 2009...

>>> bert_ama('How many carbon atoms does buckminsterfullerene have?', source='google', max_urls=5, top_n=1)
 	text 	score 	context
1 	sixty 	0.801346 	[is slippery and has a low melting point. buck...

>>> bert_ama('when is the independence day celebrated in india?', source='google', max_urls=5, top_n=1)
 	text 	score 	context
4 	15th of august 	1.086717 	[independence day of india, which is celebrate...

>>> bert_ama('what is the answer to the ultimate question of life, the universe, and everything?', source='google', max_urls=5, top_n=1)
 	text 	score 	context
1 	42 	1.347246 	[slang dictionary 42 [ fawr - tee too ] what d...
```