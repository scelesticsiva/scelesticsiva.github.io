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

Now let us first, generate some toy dataset for illustration. I will be generating 2-dimensional data from U[-1,1] and create target variables using the function $ f(\mathbf{x},\mathbf{w}) = a_0 + a_1 \mathbf{x} $ with gaussian noise with zero mean and 0.2 variance. Let us choose $ a_0 $,$ a_1 $ be 0.3,0.5 respectively.

{% highlight python %}
a_0 = -0.3
a_1 = 0.5
beta_inverse = 0.2
data = np.random.uniform(-1,1,size = [100,1])
target = (a_0 + a_1 * data) + np.random.normal(loc = 0,scale = beta_inverse,size = [100,1])
ax = sns.regplot(x = data,y = target,fit_reg = False,color = "r")
{% endhighlight %}
![synthetic_dataset](/assets/synthetic_datapoints.png){:class="img-responsive"}

We need to develop a distribution in the weights space, such that any sample from that distribution reduces the error between data and targets. We can do that by first defining prior and likelihood distributions.

The prior distribution is the distribution of the weights before seeing the data. The likelihood is the distribution of the targets given the data and weights. Let us define prior distribution to be isotropic Gaussian of the form,
$$ p( \mathbf{w}) = \mathcal{N}(\mathbf{w} \| 0,\alpha^{-1} I) $$

Let the likelihood be of the form $ p(\mathbf{t} \| \mathbf{x} , \mathbf{w}) = \prod_{i=1}^{N} \mathcal{N}(t_i \| \mathbf{w}^T x_n,\beta^-1) $. The posterior distrbution of the weights can be written as, $p(\mathbf{w} \| \mathbf{t}) \propto p(\mathbf{t} \| \mathbf{x} , \mathbf{w}) p( \mathbf{w}) $

Since both the prior and likelihood have the same gaussian distribution, the posterior is also of the form gaussian distribution and the parameters of it can be derived in closed form,[from "Pattern Recognition and Machine Learnin - C.Bishop"]

$$ \mathbf{m_n} = \mathbf{S_N}(\mathbf{S_0}^{-1}\mathbf{m_0} + \beta \Phi^T \mathbf{t}) $$ 

$$ \mathbf{S_N^{-1}} = \mathbf{S_0^{-1}} + \beta \Phi^T\Phi $$

If the prior is of the form $ \mathcal{N}(\mathbf{w} \| 0,\alpha^{-1} I) $ then the above two equations reduces to,

$$ \mathbf{m_n} = \beta \mathbf{S_N}^{-1} \Phi^T \mathbf{t} $$ 

$$ \mathbf{S_N^{-1}} = \alpha I + \beta \Phi^T\Phi $$

Now we plugin datapoint one by one into $\Phi$ and compute the mean and covariance of the posterior distribution of weights, once enough datapoints are given, the posterior distribution becomes super confident converging to a point in two dimensional space.

{% highlight python %}
alpha_inverse = 0.5
sns.set_style("whitegrid")
for i in range(len(data)):
    if i == 0:
        si = np.array([[1],[data[i]]])
        sN_inverse = (1/alpha_inverse)*np.eye(2) + (1/beta_inverse)*(si@si.T)
        mN = (1/beta_inverse)*((np.linalg.inv(sN_inverse))@(si@target[i]))
        w_posterior = np.random.multivariate_normal(mN,(np.linalg.inv(sN_inverse)),size = (1000))
        kd = sns.kdeplot(w_posterior[:,0],w_posterior[:,1],shade = False)
        kd.axes.set_ylim(-2,2)
        kd.axes.set_xlim(-1.5,1.5)
        plt.savefig("after_{0}.png".format(i))
        sns.plt.show()
        samples_w_prior = np.random.multivariate_normal(mN,(np.linalg.inv(sN_inverse)),size = (5))
        fig,ax = plt.subplots()
        sns.regplot(x = data[i],y = target[i],fit_reg = False,color = "r")
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        for w in samples_w_prior:
            prediction = w[0] + w[1]*data
            plt.plot(data,prediction)
        sns.plt.show()
    else:
        s0_inverse = copy.deepcopy(sN_inverse)
        m0 = copy.deepcopy(mN)
        si = np.array([[1],[data[i]]])
        sN_inverse = s0_inverse + (1/beta_inverse)*(si@si.T)
        mN = np.linalg.inv(sN_inverse)@((s0_inverse@m0) + ((1/beta_inverse)*(si@target[i])))
        w_posterior = np.random.multivariate_normal(mN,(np.linalg.inv(sN_inverse)),size = (1000))
        kd = sns.kdeplot(w_posterior[:,0],w_posterior[:,1],shade = False)
        kd.axes.set_ylim(-2,2)
        kd.axes.set_xlim(-1.5,1.5)
        if i == 5 or i == 20 or i == 50 or i == 100:
            sns.plt.savefig("after_{0}.png".format(i))
        sns.plt.show()
        samples_w_prior = np.random.multivariate_normal(mN,(np.linalg.inv(sN_inverse)),size = (5))
        fig,ax = plt.subplots()
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        sns.regplot(x = data[0:i],y = target[0:i],fit_reg = False,color = "r")
        for w in samples_w_prior:
            prediction = w[0] + w[1]*data
            plt.plot(data,prediction)
        if i == 5 or i == 20 or i == 50 or i == 100:
            sns.plt.savefig("lines_after_{0}.png".format(i))
        sns.plt.show()
{% endhighlight %}

For true Bayesian treatment, we should have hyper prior over the parameters $ \alpha, \beta $ but we will consider these values to be known in this example.

Firstly, the prior distribution looks like this,
![prior_distribution](/assets/prior.png){:class="img-responsive"}
After a single datapoint, the posterior distribution of the weights looks like this,
![after_single_datapoint](/assets/after_0.png){:class="img-responsive"}
![after_single_datapoint](/assets/lines_after_0.png){:class="img-responsive"}
After 5 datapoints,
![after_five_datapoints](/assets/after_5.png){:class="img-responsive"}
![after_five_datapoints](/assets/lines_after_5.png){:class="img-responsive"}
After 20 datapoints,
![after_twenty_datapoints](/assets/after_20.png){:class="img-responsive"}
![after_twenty_datapoints](/assets/lines_after_20.png){:class="img-responsive"}
After 50 datapoints,
![after_fifty_datapoints](/assets/after_50.png){:class="img-responsive"}
![after_fifty_datapoints](/assets/lines_after_50.png){:class="img-responsive"}
As you can see as the number of data points increases, the distribution becomes narrower [i.e confidence in values of weights increases] and converging to the true values of [-0.3,0.5]
