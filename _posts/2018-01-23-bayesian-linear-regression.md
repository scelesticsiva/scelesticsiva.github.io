---
layout: post
comments: true
title:  "Bayesian Linear Regression"
date:   2018-01-23 08:07:00 -0500
use_math: true
type: post
---

We all must have learnt about Linear regression, but what is Bayesian Linear Regression? It is just `Bayesian` treatment of linear regression, what do I mean by that? We consider the weights in linear regression as random variables. Now, why on earth would we do that?

Remember when you decide, weights are random you can no longer have a single value[s] of weight[s] that defines the data but we ought to have a distribution over it[them]. [just like how you cannot be certain that the next coin toss is gonna land in heads, but only associate it with a probability]

What is the purpose of this? Well...Firstly, when you decide on some value[s] of weight[s] that describe the data, after Bayesian treatment you will know how certain you are about particular weight[s], so the weight[s] not only help in prediction but also captures uncertainity in prediction.

Secondly, when you have some prior knowledge of weights [in most cases you won't], we can incorporate in the problem at hand in the form of prior distribution. We will see shortly what I mean by that.

Finally, while making predictions you are not considering a single values[s] of weights[s], but a distribution of weight[s]. You will be averaging over all the decisions that correspond to different weights weighted by their probability of occuring, this elimiates the problem of over-fitting to training data.

Now let us first, generate some toy dataset for illustration. I will be generating 2-dimensional data from U[-1,1] and create target variables using the function $ f(\mathbf{x},\mathbf{w}) = a_0 + a_1 \mathbf{x} $ with gaussian noise with zero mean and 0.1 variance. Let us choose $ a_0 $,$ a_1 $ be 0.3,0.5 respectively.

