# NLP Exercice 1

The goal of the first exercise is to implement skip-gram with negative-sampling from scratch. 

## Implementation : our thought process

The first step of our thought process is the preprocessing of the data. After loading the sentences from the file, we need to clean them in order to remove all the characters we don't want such as the dot, the #, the figures and so on. It is done in the function named cleaning.
Then, we need to build a vocabulary and tokens for each word of the vocabulary. It is done in the function vocab_ids.

Now that we have our sentences cleaned and the associated vocabulary, we can implement the skip-gram with negative sampling.
For that, we define for each word its context words namely the words that are situated in a window of two words of the center word in the sentence. The pairs (center word, context word) are positive pairs for which we assign the mark 1.
Negative sampling implies to define negative pairs as well that will be used to train the skip-gram. For each positive pair marked by 1, we add k negative pairs marked by -1. The negative pairs are computed by taking randomly a word among all the vocabulary as a context word for the center word.
Thus, we have created a cooccurences matrix with positive and negative words (function create_pairs_pos_neg).
This matrix will be used in the training of the skip-gram.




## Additional resources : web sites and papers

+ https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/ 
+ http://mediamining.univ-lyon2.fr/people/guille/word_embedding/skip_gram_with_negative_sampling.html  
+ https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb 
+ http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/ 
+ https://github.com/deborausujono/word2vecpy
+ https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
+ http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/ 


