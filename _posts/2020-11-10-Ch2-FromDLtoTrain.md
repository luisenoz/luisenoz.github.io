# From DataLoader to a Trained Model

## From Data to DataLoader
**DataLoaders** is a <ins>class</ins> that just stores whatever **DataLoader** <ins>objects</ins> we pass to it and makes them available as *train and valid*.  
It’s important in fastai because it provides the data for our model.

To turn our downloaded data into a DataLoaders object, we need to tell fastai at least four things: 
1. What kinds of data we are working with  
2. How to get the list of items  
3. How to label these items  
4. How to create the validation set

So far we have seen a number of *factory methods* for particular combinations of these things, 
which are convenient when you have an application and data structure that happen to fit into those predefined methods.  
For when you don’t have those particular combinations, fastai has an extremely flexible system called the **data block API**.  
Here is the datablock I defined for my Autsrlian places dataset:

```python
aus = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```

First we provide a tuple specifying the types we want for the independent and dependent variables: *(images for x and Categories for y)*
```python
blocks=(ImageBlock, CategoryBlock)
```
Since our underlying items will be file paths, we have to tell fastai how to get a list of those files.  
The *get_image_files* function takes a path, and returns a list of all of the images in that path (recursively, by default): 
```python
get_items=get_image_files
```

Often, datasets downloaded from internet will already have a validation set defined.  
Sometimes this is done by placing the images for the training and validation sets into different folders.  
Sometimes it is done by providing a CSV file in which each filename is listed along with which dataset it should be in.  
There are many ways that this can be done, and fastai has the flexibility to manage this by using one of the predefined classes or by writing our own.  
In this case, we want to split our training and validation sets randomly.  
However, we would like to have the same training/validation split each time we run this notebook, so we fix the random seed.
```python
splitter=RandomSplitter(valid_pct=0.2, seed=42)
```

To tell fastai what function to call to create the labels *(y)* in our dataset, we use the *parent_label* function provided by fastai 
that simply creates the label from the name of the folder a file is in.  
Because we put each of our places images into folders based on the type of place, this is going to give us the labels that we need.
```python
get_y=parent_label
```

Our images are all different sizes, and this is a problem for deep learning.  
To group them in a big array (usually called a tensor) that is going to go through our model, 
they all need to be of the same size. So, we need to add a transform that will resize these images to the same size.  
*Item transforms* are pieces of code that run on each individual item, (whether it be an image, category, or so forth).  
fastai includes many predefined transforms; and we will use the *Resize* transform here and specify a size of 128 pixels:
```python
item_tfms=Resize(128)
```

So far, we created a *DataBlock object*. And that is only like a template for creating a DataLoaders.  
We still need to tell fastai the actual source of our data — in this case, the path where the images can be found:
```python
dls = aus.dataloaders(path)
```

A DataLoader is a class that provides batches of a few items at a time to the GPU.
And when you loop through a DataLoader, fastai will give you 64 (by default) items at a time, all stacked up into a single tensor.  
We can take a look at a few of those items by calling the *show_batch* method on a DataLoader: 
```python
dls.valid.show_batch(max_n=10, nrows=1)
```
And *magically*, just by running that cell, I've got a sample of 10 beatiful images with all my 5 places represented!

By default, the *Resize* function crops the images to fit a square shape of the size requested, using the full width or height. This can result in losing some important details.   
Alternatively, you can ask fastai to pad the images with zeros (black), or squish/stretch them:
But all these approaches are somewhat wasteful or problematic:
- If we squish or stretch the images, they end up as unrealistic shapes, leading to a model that learns that things look different from how they actually are.
- If we crop the images, we could end up removing some of the features that allow us to perform recognition.
- If we pad the images, we have a whole lot of empty space, which is just wasted computation for our model and results in a lower effective resolution 
for the part of the image we actually use.

Instead, what we normally do in practice is to randomly select part of the image and then crop to just that part.  
On each epoch (which is one complete pass through all of our images in the dataset), we randomly select a different part of each image.  
This means that our model can learn to focus on, and recognize, different features in our images.  
It also reflects how images work in the real world, where different photos of the same thing may be framed in slightly different ways.

To do that, we replace *Resize* with ***RandomResizedCrop***, which is the transform that provides the behavior just described.  
The most important parameter to pass in is *min_scale*, which determines how much of the image to select at minimum each time:

```python
aus = aus.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = aus.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```
(we used unique=True to have the same image repeated 4 times with different versions of this *RandomResizedCrop* transform)

RandomResizedCrop is a specific example of a more general technique, called *data augmentation*.

## Data Augmentation

Data augmentation refers to creating random variations of our input data, such that they appear different but do not change the meaning of the data.  
Examples of common data augmentation techniques for images are rotation, flipping, perspective warping, brightness changes, and contrast changes.  
Because our images are now all the same size, we can apply these augmentations to an entire batch of them using the GPU, which will save a lot of time.  
To tell fastai we want to use these transforms on a batch, we use the *batch_tfms parameter* instead of the *RandomResizedCrop*.  
And for the same reason, we also use double the amount of augmentation compared to the default:

```python
aus = aus.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = aus.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

## Training the model and using it to clear the data

It's time to train the aus classifier.  
We don't have a lot of data, so to train our model, we’ll use *RandomResizedCrop*, an image size of 224 pixels, which is fairly standard for image classification, and the default *aug_transforms*:
```python
aus = aus.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = aus.dataloaders(path)
```

We can now create our Learner and fine-tune it, as we did in Chapter 1:
```python
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```
And these are the results:  


|0	    |2.028729   |0.747316	|0.275510	|00:15|
|-------|-----------|-----------|-----------|-----|
|epoch	|train_loss	|valid_loss	|error_rate	|time |
|0	    |0.264058	|0.162271	|0.040816	|00:15|
|1	    |0.173743	|0.068755	|0.020408	|00:15|
|2	    |0.118938	|0.038338	|0.010204	|00:16|
|3	    |0.095285	|0.034595	|0.010204	|00:16|

We trained the model in around 1 minute to an error rate of 1%!!! *(Of course I'm not going to highlight that the book's model showed a 1.6% error rate)*

To visualise the types of errors the model is doing, we can create a ***Confusion Matrix***.  
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

![confmatx](https://github.com/luisenoz/luisenoz.github.io/blob/master/images/Unknown.png)

As you can see, there was only 1 error, where the model classified a picture of Ayers Rock as one of The Great Barrier Reef.  
It’s helpful to see where exactly our errors are occurring, to see whether they’re due to a dataset problem 
(e.g., images that aren’t any of our aus places at all, or are labeled incorrectly) or a model problem 
(perhaps it isn’t handling images taken with unusual lighting, or from a different angle, etc.).  
To do this, we can sort our images by their loss.  
The loss is a number that is higher if the model is incorrect (especially if it’s also confident of its incorrect answer), 
or if it’s correct but not confident of its correct answer.  
The *plot_top_losses* function shows us the images with the highest loss in our dataset:
```python
interp.plot_top_losses(5, nrows=1)
```
The output shows that the image with the highest loss is one that has been predicted as “The Great Reef Barrier” with 67% confidence. 
However, it’s labeled as “Ayers Rock”

![highloss](https://github.com/luisenoz/luisenoz.github.io/blob/master/images/Unknown-2.png)

*What do you think? I'm sure it's not the Grea Reef Barrier, but I wouldn't bet on Ayers Rock either!*

The intuitive approach to doing data cleaning is to do it before you train a model. But a model can help you find data issues more quickly and easily.  
So, we normally prefer to train a quick and simple model first, and then use it to help us with data cleaning.
Of course, fastai includes a handy GUI for data cleaning called *ImageClassifierCleaner* that allows you to choose a category 
and the training versus validation set and view the highest-loss images (in order), 
along with menus to allow images to be selected for removal or relabeling:
```python
cleaner = ImageClassifierCleaner(learn)
cleaner
```
And then you can select any of my 5 places and see all photos one by one in the training and validation set, and relable or delete those that are incorrect.

However, *ImageClassifierCleaner* doesn’t do the deleting or changing of labels for you; 
it just returns the indices of items to change.  
So, for instance, to delete (unlink) all images selected for deletion, we would run this: 
```python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
```
To move images for which we’ve selected a different category, we would run this: 
```python
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

>Sylvain Says: Cleaning the data and getting it ready for your model are two of the biggest challenges for data scientists; 
they say it takes 90% of their time. The fastai library aims to provide tools that make it as easy as possible.

Once we’ve cleaned up our data, we can retrain our model.

Now that we have trained the model, we need to see how to deploy it to be used in practice.  
Given the extention of this post already, I think it'd be better to put the next steps into a new file.

