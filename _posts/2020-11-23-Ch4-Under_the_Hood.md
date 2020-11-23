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
