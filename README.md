# Work-from-UCLA
Takes the matrix factorization present in NMF and adds a regularization of a new matrix,
which has a correlation score of sorts between words to the loss function. 
This makes it give topics that are more contextually influenced! 
This is the math part of it as I signed an NDA so I can't get too detailed into the end-to-end pipeline.

Generally, the set up here is that you would likely have a data ingestion set up and you can take whatever
text you have from document to collection of documents. From there you should process it with sklearn 
(TfIdf, CountVectorizer, etc) and then you can use that processed matrix, as the X matrix, in this algorithm. 
Following that you should see which are the most important words in the text and generally you do that by changing 
the dimension on the matrix when you choose the dimension when creating the TfIdf settings. Following that you 
have to create the M matrix which conceptually computes a correlation type score between 2 words (For example,
if you see hot and dog together you know they're highly correlated and you're probably talking about food and 
if not it's probably a hot day and you have a pet). Next you have to create a symmetric matrix and now that I'm 
looking at the code I didn't include the code I used to do this so just know you have to and it's nontrivial as I 
took a hint for how to do this off stackoverflow. 

After you get all the matrices, you can now call semantic NMF and optimize your W,H, and S matrices. Depending on how 
you set it up, one of your W or H matrix should have the topics and words that comprise them and the other should show
which topics are found in which documents. If you want to see which words are seen in the same context window most often
you use the highest PMI words function.
