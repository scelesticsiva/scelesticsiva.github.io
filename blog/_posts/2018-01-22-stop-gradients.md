---
layout: post
comments: true
title:  "Stop gradients in Tensorflow"
date:   2018-01-22 12:19:59 -0500
excerpt: stopping gradients in tensorflow using tf.stop_gradient
keywords: tensorflow,stop,gradients,neural networks
type: post
---
This blog post is on how to use `tf.stop_gradient` to restrict the flow of gradients through certain parts of the network

There are several scenerios that may arise where you have to train a particular part of the network and keep the rest of the network in the previous state. This is when `tf.stop_gradient` comes in handy to do exactly that. Any operation that is being done inside `tf.stop_gradient` will not be updated during backpropogation.

To give some example, let us define single layer neural network with linear activations.

{% highlight python %}
x = tf.placeholder(tf.float32,[3,2])
y = tf.placeholder(tf.float32,[3,4])
w1 = tf.Variable(tf.ones([2,3]))
w2 = tf.Variable(tf.ones([3,4]))
hidden = tf.stop_gradient(tf.matmul(x,w1))
output = tf.matmul(hidden,w2)
loss = output - y
optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)
{% endhighlight %}

This is equivalent to a single hidden layer neural network with 2 input,3 hidden and 4 output units. I am using absolute error and gradient descent optimizer for demonstration purposes. For the same purpose, I have initialized the weights to be ones, so that it is clear to see the changes that happen.

Now we can run the optimizer with the following block of code and see what happens to the weights.

{% highlight python %}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("*****before gradient descent*****")
    print("w1---\n",w1.eval(),"\n","w2---\n",w2.eval())
    w1_,w2_,_ = sess.run([w1,w2,optimizer],feed_dict = {x:np.random.normal(size = (3,2)),y:np.random.normal(size = (3,4))})
    print("*****after gradient descent*****")
    print("w1---\n",w1_,"\n","w2---\n",w2_)
{% endhighlight %}
The output that we get is as follows,
{% highlight python %}
*****before gradient descent*****
w1---
 [[ 1.  1.  1.]
 [ 1.  1.  1.]] 
 w2---
 [[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]
*****after gradient descent*****
w1---
 [[ 1.  1.  1.]
 [ 1.  1.  1.]] 
 w2---
 [[ 0.67135066  0.67135066  0.67135066  0.67135066]
 [ 0.67135066  0.67135066  0.67135066  0.67135066]
 [ 0.67135066  0.67135066  0.67135066  0.67135066]]
{% endhighlight %}

As you can see, since the operation that involved w1 was inside `tf.stop_gradient`, after optimizer step only w2 got updated with the gradients and not the w1.

The full code to this demostration,

{% highlight python %}
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[3,2])
y = tf.placeholder(tf.float32,[3,4])
w1 = tf.Variable(tf.ones([2,3]))
w2 = tf.Variable(tf.ones([3,4]))
hidden = tf.stop_gradient(tf.matmul(x,w1))
output = tf.matmul(hidden,w2)
loss = output - y
optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("*****before gradient descent*****")
    print("w1---\n",w1.eval(),"\n","w2---\n",w2.eval())
    w1_,w2_,_ = sess.run([w1,w2,optimizer],feed_dict = {x:np.random.normal(size = (3,2)),y:np.random.normal(size = (3,4))})
    print("*****after gradient descent*****")
    print("w1---\n",w1_,"\n","w2---\n",w2_)
{% endhighlight %}
