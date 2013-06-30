sklearn-svm
===========

ScikitSvmPipeline implements a pipeline for processing labelled documents, classifying them with sklearn's svm, and evaluating precision and recall. For the use case of classifying whether tweets are about Hurricane Sandy, training examples that are about Hurricane Sandy are labelled positive (1) and tweets that are not are labelled negative (0). The positive training examples are in hurricanesandy_tweets.db, a collection of 88,928 tweet texts that are about Hurricane Sandy.
