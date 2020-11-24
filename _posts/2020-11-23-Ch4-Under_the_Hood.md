# UNDER THE HOOD: Training a Digit Classifier

In the book, the authors used computer vision to introduce fundamental tools and concepts for deep learning, such as:
- roles of arrays and tensors,
- broadcasting,
- stochastic gradient descent (SGD),
- learning by udpating weights automatically,
- choice of a loss fuction,
- role of minibatches.
- the maths performed by a basic neural network.   

to finally ending up puting all those pieces together.

## Pixels: The foundations of computer vision

To understand how computers handle images we’ll use one of the most famous datasets in computer vision, MNIST, 
[MINST](https://en.wikipedia.org/wiki/MNIST_database) for our experiments.

**For the initial tutorial, we are just going to try to create a model that can classify any image as a 3 or a 7.**

**First**, download a sample of MNIST that contains images of just those digits:
```python
path = untar_data(URLs.MNIST_SAMPLE)
```

We can see what’s in this directory by using *ls*, a method added by fastai.   
This method returns an object of a special fastai class called "L", which has all the same functionality of Python’s built-in list, 
plus a lot more. One of its handy features is that, when printed, it displays the count of items before listing the items themselves.   
```python
path.ls()
```
(#3) [Path('labels.csv'),Path('valid'),Path('train')]

There are 3 elements inside path. One .csv file for the labels and two directories: The MNIST dataset follows a common layout used in machine learning datasets: separate folders for the training set and the validation (and/or test) set.   
To view what’s inside the training set:
```python
(path/'train').ls()
```
(#2) [Path('train/7'),Path('train/3')].  
And "train" has two other directoires: One with the training set of 3s and another with 7s.

We can have a look in one of these folders *(using sorted to ensure we all get the same order of files)*:
```python
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
```
(#6131) [Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),
Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),
Path('train/3/10091.png')...].  
As expected, '3' is full of images of 3s; exactly 6131 of them.

We can run the followin code to look at one of the image;
```python
im3_path = threes[1]       # Create a variable with the second element in the threes list of file we created before.
im3 = Image.open(im3_path) # Open image included in the previous step using Image.open() (#)
im3.                       # Show the image
```
And we'll get an image of a handwritten number 3, taken from the MNIST dataset

*(#) We used the "Image" class from the Python Imaging Library (PIL), which is the most widely used Python package for opening, manipulating, and viewing images. 
Jupyter knows about PIL images, so it displays the image automatically.*

**Following**, as everything in computers is represented as a number, we need to convert it to a NumPy array or a PyTorch tensor to view the numbers that make up this image.   
For instance, here’s what a section of the image looks like converted to a NumPy array:
```python
array(im3)[4:10,4:10]
```
array([[  0,   0,   0,   0,   0,   0],   
       [  0,   0,   0,   0,   0,  29],   
       [  0,   0,   0,  48, 166, 224],   
       [  0,  93, 244, 249, 253, 187],   
       [  0, 107, 253, 253, 230,  48],    
       [  0,   3,  20,  20,  15,   0]], dtype=uint8). 
       
The 4:10 indicates we requested the rows from index 4 (inclusive) to 10 (noninclusive), and the same for the columns.    
NumPy indexes from top to bottom and from left to right, so since the image is 28x28, this section is located near the top-left corner of the image.

To have the same thing but as a pytorch tensor:
```python
tensor(im3)[4:10,4:10]
```
tensor([[  0,   0,   0,   0,   0,   0],   
       [  0,   0,   0,   0,   0,  29],   
       [  0,   0,   0,  48, 166, 224],   
       [  0,  93, 244, 249, 253, 187],   
       [  0, 107, 253, 253, 230,  48],    
       [  0,   3,  20,  20,  15,   0]], dtype=torch.uint8).  
       
We can slice the array to pick just the part with the top of the digit in it, and then use a Pandas DataFrame to color-code the values using a gradient, 
which shows us clearly how the image is created from the pixel values:
```python
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```
The background white pixels are stored as the number 0,   
black is the number 255,     
and shades of gray are between the two.    
The entire image contains 28 pixels across and 28 pixels down, for a total of 784 pixels.

So, now that we've seen what an image looks like to a computer, how can we go to achieve our goal of creating a model that can recognize 3s and 7s?   
The authors ask us to stop and think our own solutiona before moving forward with their solution. **Learning works best when we try to solve problems by ourselves, rather than just reading somebody else’s answers**.

## First try: Pixel similarity

The book offers a first idea: 
1. To find the average pixel value for every pixel of the 3s, then do the same for the 7s. 
2. That will give us two group averages, defining what we might call the “ideal” 3 and 7. 
3. Then, to classify an image as one digit or the other, we see which of these two ideal digits the image is most similar to.    
This certainly seems like it should be better than nothing, so it will make a good *baseline*.

> Jargon: **baseline** = A simple model that we're confident will perform resonably well, it's simple to implement and easy to test. Without a baseline model to compare, it's not going to be easy to detrmine if future super more sofisticated models are actually performing better than reasonably.   
Another good approach is to search around to find other people who have solved problems similar to ours, and download and run their code on our dataset.

<ins>Step I:</ins> Get the average of pixel values for each of our two groups.   
To create a tensor containing all the images in a directory, we will use a Python list comprehension to create a plain list of the single image tensors.   
We then check that the number of returned items seems reasonable.
```python
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)
```
(6131, 6265)

*****
> **List Comprehesion** are a powerful tool from Python, used extensively in many areas. Following is an example given in the book:    

```python
new_list = [f(o) for o in a_list if o>0]
```
This will return every element of a_list that is greater than 0, after passing it to the function f.    
It’s not only shorter to write, but also way faster than the alternative ways of creating the same list with a loop.
*****

We can also check that one of the images looks OK.    
Since we now have tensors (which Jupyter by default will print as values), rather than PIL images (which Jupyter by default will display images), 
we need to use fastai’s *show_image* function to display it:
```python
show_image(three_tensors[1])
```
We should get a good image of the digit 3.

For every pixel position, we need to calculate the average over all images of the intensity of that pixel.   
We first combine all the images in the list into a single 3 dimensional tensor *(a rank-3 tensor)*.   
Pytorch has the fucntion called *stack* that we can use to do exactly that.    
Some operations in PyTorch, such as taking a mean, require us to *cast* our integer types to float types. 
Since we’ll be needing this later, we’ll also *cast* our stacked tensor to float now.    
*Casting* in PyTorch is as simple as writing the name of the type we wish to *cast* to, *(in this case float)* and treating it as a method.    
Finally, when images are floats, the pixel values are expected to be between 0 and 1, so we will also need to divide by 255 here, *(becasue the pixels' intensity goes from 0 to 255)*.
```python
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape
```
torch.Size([6131, 28, 28])

The more important attribute of a tensor is its *shape*, that will tell us the lenght of each axis: here of 6131 images of 28 x 28 pixels.    
The *lenght* of a tensor is its *rank*, here of 3 dimensions.
```python
len(stacked_threes.shape)
```
3

We can also get a tensor’s rank directly with *ndim*:
```python
stacked_threes.ndim
```
3

Finally, we can compute our ideal 3 by taking the mean of all images along dimension 0 of the stacked rank-3 tensor. 
0 is the dimension that indexes through all images.   
For every pixel position, this will compute the average of that pixel over all images. 
The result will be one value for every pixel position, or a single image.
```python
mean3 = stacked_threes.mean(0)
show_image(mean3);
```
The ideal 3 will be very dark where all images agree and wispy and blurry where the images disagree.   

<ins>Step II:</ins> We can now pick an arbitrary 3 or 7 and meassure their *distance* from the *ideal* ones.   
- Selecting a 3 digit:
```python
a_3 = stacked_threes[1]
show_image(a_3);
```

Data scientists use two main ways to measure distance in this context: 
1. **Mean absolute difference or L1 norm**: the mean of the absolute value of differences (absolute value is the function that replaces negative values with positive values). 
2. **Root mean squared error (RMSE) or L2 norm**: the mean of the square of differences (which makes everything positive) and then take the square root (which undoes the squaring).

```python
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr
```
(tensor(0.1114), tensor(0.2021))

```python
dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr
```
(tensor(0.1586), tensor(0.3021))

In both cases, the distance between our 3 and the “ideal” 3 is less than the distance to the ideal 7, 
so our simple model will give the right prediction in this case.

*PyTorch already provides both of these as loss functions. We can find these inside torch.nn.functional, which the PyTorch team recommends importing as F (and is available by default under that name in fastai):*

```python
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
```
(tensor(0.1586), tensor(0.3021))

After so many calculations, the book moves to show us the characteristics and differences of two important mathematical structures: *NumPy arrays* and *PyTorch tensors*.   
Let's see that in the next post.
