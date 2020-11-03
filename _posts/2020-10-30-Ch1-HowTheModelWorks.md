# How the Image Recognizer Works

The chapter continues by providing a more detailed intro to what each line of the model works. But it also promises full details in later chapters. 
Therefore, I'm not going to review each line and the descriptions, but only those points or tips that took my attention.

- Normally, it's not good practice to import a whole library *(using the import * syntax)* because in large software projects it can cause problems. 
However, according to the authors,for interactive work such as in a Jupyter notebook, it works great, and the fastai library is specially designed to support this kind of interactive use, and it will import only the necessary pieces into the environment.

- Another highlight is how fastai doesn’t just return a string containing the path to the dataset, but a ***Path object***.  
Apparently, this is a really useful class from the Python 3 standard library that makes accessing files and directories much easier.  
I've never use it before, so I'd have a look at its documentation.

- It seems that computer vision datasets are normally structured in such a way that the label for an image is part of the filename or path.  
Most commonly the parent folder name.

It's well known to anyone that at least started reading about ML that ML includes two main types of models:
- **Classification models:** attempt to predict a class, or category, *(like cat or dog)*
- **Regression models:** attempt to predict one or more numeric quantities, *(like temperature, or the value of a stock)*.

## Overfitting

One of the most important parameters in the model is *valid_pct=0.2*.  
That parameter tells fastai to hold out 20% of the data and not use it for training the model at all.  
This 20% of the data is called the **validation set**; the remaining 80% is called the **training set**.  
**The validation set is used to measure the accuracy of the model.**. 

And that is absolutely critical, because if you train a model for a long enough time, it will eventually memorize the label of every item in your dataset!  
The longer you train a model, the better your accuracy will get on the training set.  
While validation set accuracy will also improve for a while, it will eventually start getting worse as the model starts to memorize the training set 
rather than finding generalizable underlying patterns in the data.  
When this happens, we say that the model is **overfitting**.

According to Jeremy's experiece, **overfitting is the single most important and challenging issue**
when training for all machine learning practitioners, and all algorithms.  
<code>It is easy to create a model that makes great predictions on the exact data it has been trained on, 
but it is much harder to make accurate predictions on data the model has never seen before.</code><sup>[(1)](#myfootnote1)</sup>

There are many methods to avoid overfitting. However, we **should use those methods only after we have confirmed that overfitting is occurring**
(i.e., if we have observed the validation accuracy getting worse during training).

## Architectures

There are many architectures available and you can even create your own; however, picking an architecture isn’t a very important part of the deep learning process.  
There are some standard architectures that work most of the time

## Metric & Loss

**Do not confuse *metric* with *loss***. 
- A metric is defined for human consumption. A a good metric is one that is easy for us to understand, 
and that shows how far or close the model is from the way you want it to perform.
- Loss instead, is a “measure of performance” that the training system can use to update weights automatically (e.g. SGD).

## Pretrained models

A model that has weights that have already been trained on another dataset is called a **pretrained model**. 
It's suggested that we should nearly always use a pretrained model, because it means that our model, before ww’ve even shown it any of your data, 
is already very capable.  
When using a pretrained model, the fastai function *cnn_learner* will remove the last layer, 
since that is always specifically customized to the original training task,
and replace it with one or more new layers with randomized weights, of an appropriate size for the dataset we are working with. This last part of the model is known as ***the head***.  

<code>**Using pretrained models is the most important method we have to allow us to train more accurate models, 
more quickly, with less data and less time and money.**</code>.  

Using a pretrained model for a task different from what it was originally trained for is known as ***transfer learning***.

So far, we have the architecture, but that only describes a template for a mathematical function; 
it doesn’t actually do anything until we provide values for the millions of parameters it contains.  
To fit a model, we have to provide at least one piece of information: how many times to look at each image (known as number of *epochs*).  
When we start with a pretrained model, we don’t want to throw away all those capabilities that it already has.  
So we don't need to ***fit*** the model but apply a process ***called fine-tuning***.  
- **Fine-Tuning:** A transfer learning technique that updates the parameters of a pretrained model by training for additional epochs 
using a different task from that used for pretraining.

## Is DL a blackbox?

Although many people complain that deep learning results in impenetrable “black box” models 
(that is, something that gives predictions but that no one can understand); according to Jeremy, this really couldn’t be further from the truth.
They suggest to read *“Visualizing and Understanding Convolutional Networks”, published in 2013, by PhD student Matt Zeiler and his supervisor, Rob Fergus.
This work showed how to visualize the neural network weights learned in each layer of a model.

## Image Recognizers Can Tackle Non-Image Tasks

An image recognizer can, as its name suggests, only recognize images. 
But a lot of things can be represented as images, which means that an image recognizer can learn to complete many tasks.

- A sound can be converted to a spectrogram, which is a chart that shows the amount of each frequency at each time in an audio file.
- A time series can easily be converted into an image by simply plotting the time series on a graph.
- Various other transformations are available for time series data. For instance, using a technique called **Gramian Angular Difference Field (GADF)**
- Using a dataset of users’ mouse movements and clicks, a fastai student turned these into pictures by drawing an image displaying the position, 
speed, and acceleration of the mouse pointer by using colored lines, and the clicks were displayed using small colored circles.

<code>A good rule of thumb for converting a dataset into an image representation: If the human eye can recognize categories from the images, 
then a deep learning model should be able to do so too.</code>

## Deep Learning Is Not Just for Image Classification

- Localizing object in a picture. In particular, creating a model that can recognize the content of every individual pixel in an image is called ***segmentation***.
- Natural language processing (NLP).
- Building models from plain tabular data.
- Recommendation systems

**TIP:** The world’s top practitioners do most of their experimentation and prototyping with subsets of their data, 
and use the full dataset only when they have a good understanding of what they have to do.

## Validation and Test sets

The goal of a model is to make predictions about data.  
Our first step was to split our dataset into two sets: 
- **the training set** (which our model sees in training) and
- **the validation set**, also known as the development set (which is used only for evaluation). 
This division of the data lets us test that the model learns lessons from the training data that generalize to new data, the validation data.  

We don’t want our model to get good results by *“cheating.”*   
If it makes an accurate prediction for a data item, that should be because it has learned characteristics of that kind of item, 
and not because the model has been shaped by actually having *seen* that particular item.  
But the situation is more subtle. In realistic scenarios we rarely build a model just by training its parameters once. 
Instead, we are likely to explore many versions of a model through various modeling choices regarding network architecture, learning rates, 
and other *hyperparameters*.  

Just as the automatic training process is in danger of overfitting the training data, we are in danger of overfitting the validation data 
through human trial and error and exploration.  
The solution is to introduce another level of even more highly reserved data: ***the test set***.  
We must hold back the test set data even from ourselves. It cannot be used to improve the model; 
it can be used only to evaluate the model at the very end of our efforts.  

<code> Training data is fully exposed, the Validation data is less exposed, and Test data is totally hidden.</code>.   

All this 3 sets of data may seem a bit extreme, but it is often necessary because models tend to gravitate toward the simplest way to do good predictions 
(memorization), and we as fallible humans tend to gravitate toward fooling ourselves about how well our models are performing. 
The discipline of the test set helps us keep ourselves intellectually honest.  

To do a good job of defining a validation set (and possibly a test set), you will sometimes want to do more than just randomly 
grab a fraction of your original dataset.  
- Remember: a key property of the validation and test sets is that they must be **representative of the new data you will see in the future**.  










<a name="myfootnote1">(1)</a>: Please, be aware that I'm uasing different text styles in this file, most of the time 
because I really think the concepts are important, but other times it's just to practice different techniques for wrinting in Markdown. 
And by the way, that's another small usefull toolkit to learn.
