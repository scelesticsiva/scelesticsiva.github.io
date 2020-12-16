---
layout: post
comments: true
title: "Finding lottery tickets, the Bayesian way"
type: post
use_math: true
---

#### Introduction
----
Lottery ticket hypothesis \cite{frankle2018lottery} talks about the existence of subnetworks in a large neural network that is responsible for the performance of the overparametrized neural network. When this subnetwork (called lottery ticket) is trained in isolation to the same number of epochs, it reaches the same accuracy (or sometimes higher accuracy) than the bigger network it was originally a part of. This gives a clue as to why do neural networks need to be overparametrized. The probability of finding these subnetworks increase exponentially with the size of the network and therefore the chance of finding one of these lucky subnetworks.

The strategy used in the original paper to identify these subnetworks is to look at the absolute magnitude of the final weights of a trained bigger network. Certain top-k percentage of absolute weights at each layer are retained and therefore form the weights of the subnetwork. While there have been several schemes explored in \cite{zhou2019deconstructing} to identify the winning tickets, I was particularly interested in using uncertainty to find these subnetworks. What would be really nice is to find these subnetworks during start of training thereby reducing significant computational overhead.

The reader should note that, this is not something one would do in practice even if finding subnetworks through unceratinty works really well, since getting uncertainty estimates of the weights can be computationally expensive than training a regular neural network. With that said, there is a possibility that when an approximate posterior distribution of the weights, obtained through a subset of data can be used to find the subnetwork for the whole dataset (although this is left for future work).

#### Bayesian Neural Networks
-----

Bayesian Neural Networks are variants of neural networks with weights ($\mathbf{W}$) treated as random variables instead of deterministic values. When we have probability distribution over the weights of a neural network, training the network on data ($\mathcal{D}$) means, finding the posterior distribution of the weights $p(\mathbf{W} | \mathcal{D})$. Finding the posterior distribution of weights is not possible most of the time due to the presence of normalization constant. 


$$    p(\mathbf{W} | \mathcal{D}) = \frac{p(\mathcal{D} | \mathbf{W}) p(\mathbf{W})}{\sum_{\mathbf{W}}p(\mathcal{D}, \mathbf{W})} $$


Variational inference is an approximate inference method to determine posterior distribution by casting it as an optimization problem. Varitional inference finds the "closest" distribution in terms of $KL$-divergence from a variational familiy of distributions. In addition, since reducing $KL$-divergence means, determining the posterior distribution itself, variational inference methods usually optimize the lower bound called \textit{Evidence Lower BOund (ELBO)} on the data-likelihood ($p(\mathcal{D}))$. Once the ELBO is optimized to convergence, the resulting variational distribution can be used in place of posterior distribution.

I used tensorflow probability to train a BNN using "DenseFlipout" layers which uses mean-field variational inference internally along with Flipout \cite{wen2018flipout} estimator for lower variance in estimating the gradients of \textit{ELBO} with respect parameters of the variational distribution. 

Once a BNN is trained, we would have mean and variance of the posterior distribution of each value of weights in the trained network. Estimates of variance of the posterior distribution of weights are referred to as the epistemic uncertainty.

#### Can uncertainty in weights be used to find lottery tickers?
-----

I wanted to answer the question, can the variance of the posterior distribution of weights of a neural network trained as a bayesian neural network be used to find the lottery ticket? 

I trained a fully connected BNN of architecture 300-100-10 on MNIST. Once I get the initial and final parameters of the variational distribution of weights, I train two deterministic neural networks of the same model architecture with same set of hyperparameters, 1) Weights starting from the initial means of the bayesian neural network 2) Weights starting from the initial means of the bayesian neural network, but with weights greater than a certain threshold on variance of the final BNN masked. When these networks are trained till convergence, this is how they trained.