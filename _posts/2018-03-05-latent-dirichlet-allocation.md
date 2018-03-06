---
layout: post
comments: true
title: "Latent Dirichlet Allocation"
type: post
use_math: true
---

This will be a two part blog post on Latent Dirichlet Allocation(LDA), one of the widely used techniques for topic modelling. In this first part, I will be talking about how LDA models a corpus and what assumptions are made during this modelling. I will leave the inference and parameter estimation to the second part. Since a lot of libraries exist for inference and parameter estimation, this will be an ideal post to get some insight for those who have been using LDA as a black box. We will see LDA in the context of topic modelling.

LDA is,
* generative - models the distribution of topics
* probabilistic - there are no hard decisions that this particular document belongs to a particular topic but a document belongs to every single topic with a probability
* three-level hierarchial Bayesian - this requires a dive into the actual model itself.

Let us define some notation and terms before we move forward,
* A word $w$ is a one hot vector whose length is the cardinality of the set V which represents the vocabulary , $w^v = 1$ and $w^u = 0$ for all $u \neq v$.
* A document is a sequence of $N$ words denoted by $\mathbf{w} = (w_1,w_2,...,w_N)$ where $w_n$ is the $n^{th}$ word in the document.
* A corpus is a collection of $M$ documents denoted by $D = \{\mathbf{w_1},\mathbf{w_2},...,\mathbf{w_M}\}$
