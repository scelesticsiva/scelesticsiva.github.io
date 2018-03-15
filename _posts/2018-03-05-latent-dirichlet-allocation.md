---
layout: post
comments: true
title: "Latent Dirichlet Allocation"
type: post
use_math: true
---

This will be a two part blog post on Latent Dirichlet Allocation(LDA), one of the widely used techniques for topic modelling. In this first part, I will be talking about how LDA models a corpus and what assumptions are made during this modelling. I will leave the inference and parameter estimation to the second part. Since a lot of libraries exist for inference and parameter estimation, this will be an ideal post to get some insight for those who have been using LDA as a black box. Though there are many use cases to LDA, we will see it in the context of topic modelling.

LDA is,
* generative - models the distribution of topics
* probabilistic - there are no hard decisions that this particular document belongs to a particular topic but a document belongs to every single topic with a probability
* three-level hierarchial Bayesian - this requires a dive into the actual model itself.

Let us define some notation and terms before we move forward,
* A word $w$ is a one hot vector whose length is the cardinality of the set V which represents the vocabulary , $w^v = 1$ and $w^u = 0$ for all $u \neq v$.
* A document is a sequence of $N$ words denoted by $\mathbf{w} = (w_1,w_2,...,w_N)$ where $w_n$ is the $n^{th}$ word in the document.
* A corpus is a collection of $M$ documents denoted by $D = \mathbf{w_1},\mathbf{w_2},...,\mathbf{w_M}$

LDA is three-level hierarchial Bayesian because it models a corpus in the following way,
* Choose a __distribution over distribution__ of topics.
* Choose a topic from the chosen distribution of topics.
* Choose a word from the conditional distribution of words given the chosen topic.

![LDA_generative_process](/assets/LDA_generative_process.png){:class="img-responsive"}

In case of LDA, the __distribution over distribution__ is a dirichlet distribution parameterized by $\alpha$
[This is a [great article](http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/) to visualize dirichlet distribution and I will be using the code from the post here]
Each point sampled from a dirichlet distribution is a valid multinomial distribution and it is considered to be distribution over the topics. Depending on the topic chosen, the words are sampled from another multinomial distribution.

In other words,
* $\theta_d \sim Dir(\alpha)$
* $z_n \sim Multinomial(\theta_d)$
* $w_n \sim Multinomial(\beta_{z_n})$

Here $z_n$ denotes the topic for each word and the last multinomial distribution is parameterized by $\beta$ depending on the chosen topic, therefore indicated as $\beta_{z_n}$. As you can probably realise by now $\mathbf{\beta}$ is an array consisting of multinomial distributions for each topic.

Let us see some examples, our corpus contains the following latent topics and words
{% highlight python %}
TOPICS = np.array(["sports","business","politics"])
WORDS = np.array(["stock","football","vote","score"])
{% endhighlight %}

We first sample a multinomial distribution for a document from a dirichlet distribution parameterized by $\alpha$
{% highlight python %}
NO_TOPICS = 3
NO_WORDS = 4
"""
first select a multinomial distribution from a dirichlet distribution
"""
alpha = [0.999,0.999,0.999]
theta = np.random.dirichlet(alpha,size = 1)  
print(theta)

[ 0.47742552  0.49092008  0.0316544 ]
{% endhighlight %}

Then we will sample a topic for a word in the document
{% highlight python %}
"""
select a topic from the multinomial distribution
"""
z = np.random.multinomial(1,theta[0])
print("TOPIC CHOSEN:",TOPICS[np.where(z == 1)])

TOPIC CHOSEN: ['sports']
{% endhighlight %}

Before gonig to the next step,we will define the $\beta$ matrix, each row corresponds to a multinomial distribution of words,
{% highlight python %}
"""
Choose a multinomial distribution depending on the topic chosen
"""
topic_word_matrix = np.zeros((NO_TOPICS,NO_WORDS))
topic_word_matrix[0,:] = np.array([0.05,0.55,0.05,0.35]) # "football" and "score" are more probable in the topic sports
topic_word_matrix[1,:] = np.array([0.75,0.05,0.15,0.05]) # "stock" is more probable in the topic business
topic_word_matrix[2,:] = np.array([0.15,0.15,0.55,0.15]) # "vote" is more probable in the topic politics
print(np.sum(topic_word_matrix,axis = 1))

[ 1.  1.  1.]
{% endhighlight %}

After that, we will sample words from the chosen multinomial disrtibution of words,
{% highlight python %}
"""
Finally sample the words from the chosen multinomial distribution
"""
words = np.random.multinomial(1,topic_word_matrix[np.where(z == 1)][0])
print("WORD CHOSEN:",WORDS[np.where(words == 1)])

WORD CHOSEN: ['football']
{% endhighlight %}

This is how the generative process of LDA models the corpus. We will see how to estimate the parameters $\alpha,\beta,\theta$ in the next post.
