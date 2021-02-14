# MLQA


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
