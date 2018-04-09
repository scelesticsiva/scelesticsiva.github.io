---
layout: post
comments: true
title: "Why Naive Bayes is Naive?"
type: post
use_math: true
---

Naive Bayes Classifier is a widely used classifier because of its simplicity and ease of implementation. We cannot disregard this classifier as being "naive" and cannot be used for complicated tasks. So, if this classifier is powerful enough what is so naive about it?

It is naive because the classifier assumes that the attributes of the data are independent to each other. This might be true in some cases depending on the data we are dealing with.

Let us choose a well known Titanic dataset(from here) and extract only the columns "Sex","Pclass","Fare" and the label "Survived" as our example.

Our attributes(features in mahcine learning terminology) are sex,pclass and fare which we will be using to predict the label. The important assumption that the Naive Bayes classifer makes while modelling the data is that these attributes are independent given the labels.

In other words,
$p(x_1,x_2,...,x_d|C_k) = p(C_k)\prod_{i = 1}^{n}p(x_i|C_k)$

For our particular dataset, that looks something like this,
$p(Sex,Pclass,Fare|survived/dead) = p(survived/dead) * p(Sex|survived/dead) $
						$* p(Pclass|survived/dead) * p(Fare|survived/dead)$