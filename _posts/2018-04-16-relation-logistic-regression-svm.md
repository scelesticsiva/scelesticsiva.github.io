---
layout: post
comments: true
title:  "Logistic regression and SVM"
date:   2018-04-16 19:15:59 -0500
excerpt: Only the loss is different in logistic regression and SVM
keywords: logistic regression,SVM,hinge loss,logistic loss
type: post
---

In this post we will see how logistic regression and SVM(Support Vector Machines) are related. Ever since I started learning about machine
learning, I wondered how several seemingly unrelated things in ML are well connected to each other. This was one of the relations that I learned during my
early days of learning ML.

Before going into the details, let me state in plain english what is it we are going to see in this blog,

```
Logistic regression and SVM are different flavours of the same 'thing'.
```

Before going into what that `thing` is, we will first see what actually `learning` means in machine learning sense.

# What is Learning?

Let us define our data as follows (we will consider a supervised learning problem here),

$$ {(\mathbf{x_i}, \mathbf{y_i}), ... ,(\mathbf{x_n}, \mathbf{y_n})\} $$

We want to learn a function that is mapping from input $ \mathbf{x_i} $ to $ \mathbf{y_i} $, such that we give out a reasonable prediction even
when a new datapoint - out of training set - is given to us. Formally, we want to find a function which does the following,

$$ min_{f \in \mathcal{F}} \quad \mathcal{E}(f) ; \mathcal{E}(f) = \mathbb{E}_{p_{(X,Y)}}\mathcal{L}(f(\mathbf{x}), \mathbf{y}) $$

Let us break this equation down, $ \mathcal{L}(.,.) $ (otherwise known as loss function) is a measure of how good we are predicting $ \hat{\mathbf{y}} $ when compared to the actual labels.
When this `loss` is averaged over all datapoints of $ (\mathbf{x_i}, \mathbf{y_i}) $ weighted by their probabilities, we get the expected risk $ \mathcal{E} $. We
want to find a function from a family of functions $ \mathcal{F} $ that minimizes the expected risk.

But remember, we have access to only the datapoints from $ p_{(X,Y) $,
so we approximate the above formulation by its empirical version as follows,

$$ min_{f \in \mathcal{F}} \quad \hat{\mathcal{E}}(f); \hat{\mathcal{E}}(f) = \sum_{i=1}^{n}\mathcal{L}(f(\mathbf{x}), \mathbf{y}) $$

This is called as Empirical Risk Minimization. Therefore, to start learning from data we need to define two things,

* Space of functions we would search for our true function 
* Loss function to determine how well we are doing given a function

Coming back to the connection between Logistic regression and SVM, they both are flavours of ERM with space of functions being the same i.e linear (excluding kernel versions of Logistic regression
and SVM) and they differ only by how they define their losses. Logistic regression uses logistic loss $ \mathcal{L}_{logistic} = ln(1 + exp(-\mathbf{y}f(\mathbf{x}))) $ and 
linear SVM uses hinge loss $ \mathcal{L}_{SVM} = [1 - \mathbf{y}f(\mathbf{x})]_{+} $. They both can be considered as continuous approximations
of misclassification error.


# Example

## Setting up a toy dataset

{% highlight python %}
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    NO_OF_DATAPOINTS = 200
    
    mu_1 = [-4,-3]
    sigma_1 = [[3, 0], [0, 3]]
    
    mu_2 = [4,3]
    sigma_2 = [[5, 0], [0, 5]]
    
    np.random.seed(99)
    data_1 = np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=NO_OF_DATAPOINTS//2)
    labels_1 = np.ones((len(data_1), 1))
    data_2 = np.random.multivariate_normal(mean=mu_2, cov=sigma_2, size=NO_OF_DATAPOINTS//2)
    labels_2 = np.zeros((len(data_2), 1))
    data = np.vstack((data_1, data_2))
    labels = np.vstack((labels_1, labels_2))
    labels_colored = np.chararray((len(labels), 1), unicode=True)
    labels_colored[labels == 0] = "g"
    labels_colored[labels == 1] = "k"
    
    plt.scatter(data[:, 0], data[:, 1], alpha = 0.5, c=labels_colored[:,0])
{% endhighlight %}
![synthetic_dataset](/assets/data_points_LR_SVM.png){:class="img-responsive"}

## Logistic Regression

{% highlight python %}
    def logistic_regression():
        logistic_x = tf.placeholder(tf.float32, [None, 2])
        logistic_y = tf.placeholder(tf.float32, [None, 1])
        logistic_lr = tf.placeholder(tf.float32)
        
        with tf.variable_scope("logistic"):
            w = tf.get_variable(name="logistic_w", shape=[2,1], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="logistic_b", shape=[1], initializer=tf.constant_initializer(0.01))
            
        logits_ = tf.matmul(logistic_x, w) + b
        y_hat = tf.nn.sigmoid(logits_)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_, labels=logistic_y))
        
        optimizer = tf.train.GradientDescentOptimizer(logistic_lr).minimize(loss)
        
        return loss, optimizer, w, b, logits_, y_hat, logistic_x, logistic_y, logistic_lr
        
    # Start training
    logistic_loss, logistic_op, logistic_w, logistic_b, logistic_logits, logistic_y_hat, x_placeholder, y_placeholder, lr_placeholder = logistic_regression()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for e in range(10000):
            _, loss_ = sess.run([logistic_op, logistic_loss],  feed_dict={x_placeholder: data,
                                                                         y_placeholder: labels,
                                                                         lr_placeholder: 10})
            if e % 1000 == 0:
                print(loss_)
        predictions_logistic = sess.run(logistic_y_hat, feed_dict={x_placeholder: points_for_decision_boundary,
                                                         y_placeholder: np.zeros((len(points_for_decision_boundary), 1))})
{% endhighlight %}

## Linear - SVM

{% highlight python %}
    def SVM():
        svm_x = tf.placeholder(tf.float32, [None, 2])
        svm_y = tf.placeholder(tf.float32, [None, 1])
        svm_lr = tf.placeholder(tf.float32)
        
        with tf.variable_scope("svm"):
            w = tf.get_variable(name="svm_w", shape=[2,1], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="svm_b", shape=[1], initializer=tf.constant_initializer(0.01))
            
        logits_ = tf.matmul(svm_x, w) + b
        loss = tf.reduce_mean(tf.losses.hinge_loss(logits=logits_, labels=svm_y))
        
        optimizer = tf.train.GradientDescentOptimizer(svm_lr).minimize(loss)
        
        return loss, optimizer, w, b, logits_, svm_x, svm_y, svm_lr
        
    # Start training
    svm_loss, svm_op, svm_w, svm_b, svm_logits, x_placeholder_svm, y_placeholder_svm, lr_placeholder_svm = SVM()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for e in range(10000):
            _, loss_ = sess.run([svm_op, svm_loss],  feed_dict={x_placeholder_svm: data,
                                                                y_placeholder_svm: labels,
                                                                lr_placeholder_svm: 10})
            if e % 1000 == 0:
                print(loss_)
        predictions_SVM = sess.run(svm_logits, feed_dict={x_placeholder_svm: points_for_decision_boundary,
                                                      y_placeholder_svm: np.zeros((len(points_for_decision_boundary), 1))})
{% endhighlight %}

## Plotting decision boundary

{% highlight python %}

def plot_decision_boundary(predictions, greater_than = 0.5):
    thresholded_predictions = (predictions > greater_than).astype(np.int8)
    label_color = np.chararray((len(thresholded_predictions), 1), unicode=True)
    label_color[thresholded_predictions == 0] = "r"
    label_color[thresholded_predictions == 1] = "b"

    plt.scatter(points_for_decision_boundary[:, 0], points_for_decision_boundary[:, 1], c=label_color[:, 0], alpha = 0.02)
    plt.scatter(data[:, 0], data[:, 1], alpha = 0.8, c=labels_colored[:, 0])

"""
Setting up the grid
"""

GRID_SIZE = 0.08
x_min, x_max = data[:, 0].min() - GRID_SIZE, data[:, 0].max() + GRID_SIZE
y_min, y_max = data[:, 1].min() - GRID_SIZE, data[:, 1].max() + GRID_SIZE

x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, GRID_SIZE), np.arange(y_min, y_max, GRID_SIZE))

x_ravel, y_ravel = np.expand_dims(x_grid.ravel(), axis=-1), np.expand_dims(y_grid.ravel(), axis=-1)

points_for_decision_boundary = np.hstack((x_ravel, y_ravel))
print(points_for_decision_boundary.shape)
    
    
plot_decision_boundary(predictions_logistic)
plot_decision_boundary(predictions_SVM, greater_than=0)

{% endhighlight %}
Decision boundary for Logistic regression,
![synthetic_dataset](/assets/decision_boundary_LR.png){:class="img-responsive"}
Decision boundary for linear SVM,
![synthetic_dataset](/assets/decision_boundary_SVM.png){:class="img-responsive"}


