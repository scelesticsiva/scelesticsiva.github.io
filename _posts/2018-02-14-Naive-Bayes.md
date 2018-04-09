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

Now, since "Fare" feature in our dataset is real valued, we have to define a continuous distribution over it. Let us take a simple Gaussian distribution assumption and calculate the mean and variance of "Fare" for both survived and dead people.
<center>$p(Fare|survived) = \mathcal{N}(\mu_s,\sigma_s)$</center>
<br>
where $\mu_s = \frac{1}{N}\sum Fare Of Survived$ is the average fare for survived people and $\sigma_s = \sqrt{\frac{\sum (Fare-\mu_s)^2}{N}}$ is the standard deviation of the fare of survived people. We can similarly calculate $p(Fare|dead)$ my taking the average($\mu_d$) and standard deviation($\sigma_d$) of fare of dead people.

With all those probabilities calculated, we can start predicting whether a person has survived or dead given a person's Sex,Fare and Pclass.
<center> $prediction = argmax(p(Sex_{test},Pclass_{test},Fare_{test}|survived),p(Sex_{test},Pclass_{test},Fare_{test}|dead))$</center>

The entire code for calculation of probabilities and prediction is as follows,
{% highlight python %}
data = pd.read_csv(DATASET_PATH,index_col = False)
example_data = data[["Sex","Fare","Pclass","Survived"]]
train_data,test_data = train_test_split(example_data,test_size = 0.20) #Splitting the dataset into testing and training
train_data.head(10)

survivors_data = train_data[train_data["Survived"] == 1]
total_survivors = len(survived_data)

dead_people_data = train_data[train_data["Survived"] == 0]
total_dead = len(dead_people_data)

p_survived_sex = {"male":0,"female":0}
p_dead_sex = {"male":0,"female":0}
p_survived_sex["male"] = len(survivors_data[survivors_data["Sex"] == "male"])/total_survivors
p_survived_sex["female"] = len(survivors_data[survivors_data["Sex"] == "female"])/total_survivors
p_dead_sex["male"] = len(dead_people_data[dead_people_data["Sex"] == "male"])/total_dead
p_dead_sex["female"] = len(dead_people_data[dead_people_data["Sex"] == "female"])/total_dead

p_survived_class = {1:0,2:0,3:0}
p_dead_class = {1:0,2:0,3:0}
for i in range(1,4):
    p_survived_class[i] = len(survivors_data[survivors_data["Pclass"] == i])/total_survivors
    p_dead_class[i] = len(dead_people_data[dead_people_data["Pclass"] == i])/total_dead

p_survived_fare_mean,p_survived_fare_std = np.mean(survivors_data["Fare"]),np.std(survivors_data["Fare"])
p_dead_fare_mean,p_dead_fare_std = np.mean(dead_people_data["Fare"]),np.std(dead_people_data["Fare"])

predictions = []
for index,each in test_data.iterrows():
    p_survived = (total_survivors/len(train_data)) * p_survived_sex[each["Sex"]] * p_survived_class[int(each["Pclass"])] * \
            norm.pdf(each["Fare"],p_survived_fare_mean,p_survived_fare_std)
    p_dead = (total_dead/len(train_data)) * p_dead_sex[each["Sex"]] * p_dead_class[each["Pclass"]]*\
                norm.pdf(each["Fare"],p_dead_fare_mean,p_dead_fare_std)
    predictions.append(int(p_survived>p_dead))

print("Accuracy:",np.mean(np.equal(predictions,test_data["Survived"])))
{% endhighlight %}

