---
layout: post
comments: true
title: "Why Naive Bayes is Naive?"
type: post
use_math: true
---

Naive Bayes Classifier is a widely used classifier because of its simplicity and ease of implementation. We cannot disregard this classifier as being "naive" and cannot be used for complicated tasks. So, if this classifier is powerful enough what is so naive about it?

It is naive because the classifier assumes that the attributes of the data are independent to each other. This might be true in some cases depending on the data we are dealing with.

Let us choose a well known Titanic dataset(from here) and extract only the columns "Sex","Pclass","Fare" and labels "Survived" for  illustration. Here is a snippet of it.

 ![Titanic_dataset](/assets/titanic_dataset.png){:class="img-responsive"}

Our attributes(features in mahcine learning terminology) are sex,pclass and fare which we will be using to predict the label. The important assumption that the Naive Bayes classifer makes while modelling the data is that these attributes are independent given the labels.

In other words,
<center> $p(x_1,x_2,...,x_d|C_k) = p(C_k)\prod_{i = 1}^{n}p(x_i|C_k)$ </center>
<br>

For our particular dataset, that looks something like this,
<br>
<center>$p(Sex,Pclass,Fare|survived/dead) = p(survived/dead) * p(Sex|survived/dead) $
						$ * p(Pclass|survived/dead) * p(Fare|survived/dead)$ </center>
<br>

Let us see how each of those terms in the above equation can be calculated,
<center> $p(survived) = \frac{No. of suvivors}{Total No. of people}$ </center>
similarly,
<center> $p(dead) = \frac{No. of dead}{Total No. of people}$ </center>

Since Sex and Pclass are categorical and ordinal variables respectively, we can calculate those probabilities as follows,
<center> $p(male|survived) = \frac{No. of Survived Who Are Male}{Total No. of Survived People}$ </center>
<br>
<center> $p(female|survived) = \frac{No. of Survived Who Are Female}{Total No. of Survived People}$ </center>
<br>
The same can be calculated for dead people as well.

In the same way,
<center> $p(Pclass = 1|survived) = \frac{No. of People Travelled In 1st Class Who Survived}{Total No. of Survived People}$ </center>
<br>
Likewise, you can calculate probability for each class for both survived and dead people.
