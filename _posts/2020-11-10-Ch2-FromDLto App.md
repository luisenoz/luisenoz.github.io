# From DataLoader to Application

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

