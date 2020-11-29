## NumPy arrays and PyTorch Tensors

NumPy is the most widely used library for scientific and numeric programming in Python. However, unlike PyTorch tensors, 
it does not support using the GPU or calculating gradients, which are both critical for deep learning.    
*(fastai added some other features to NumPy and PyTorch, so if you find that any code in the book doesn't work in your computer,
it's possible you forgot to add the following line at the start of your notebook: ```from fastai.vision.all import *```)*

Python is slow when compared with many other languages. Anything fast in Python, NumPy, or PyTorch is likely to be a wrapper for a compiled object 
written (and optimized) in another language — specifically, C.    

A **NumPy array** is a multidimensional table of data, with <ins>all items of the same type</ins>.    
NumPy will store all items as a compact C data structure in memory and has a wide variety of operators and methods that can run computations 
on these compact structures at the same speed as optimized C, because they are written in optimized C.

A **PyTorch tensor** is nearly the same thing as a NumPy array, but with an additional restriction that unlocks additional capabilities.    
It is a multidimensional table of data, with all items of the same type, but with the restriction that a tensor cannot use just any old type
— <ins>it has to use a single basic numeric type for all components</ins>.    
A PyTorch tensor cannot be *jagged* (like arrays of arrays, where the innermost arrays are of different sizes). 
<ins>A tensor is always a regularly shaped multidimensional rectangular structure.</ins>    
The vast majority of methods and operators supported by NumPy are also supported by PyTorch, 
but PyTorch tensors have additional capabilities.    
One major capability is that <ins>these structures can live on the GPU</ins>, in which case their computation will be optimized for the GPU and can run much faster.    
In addition, <ins>PyTorch can automatically calculate derivatives</ins> of these operations, including combinations of operations. 
It would be <ins>impossible to do deep learning in practice without this capability</ins>.

> C is a low-level *(low-level means more similar to the language that computers use internally)* language that is very fast compared to Python.    
To take advantage of its speed while programming in Python, try to avoid as much as possible writing loops, 
and replace them by commands that work directly on arrays or tensors.

The authors think that perhaps the most important new coding skill for a Python programmer to learn is how to effectively use the array/tensor APIs.    
and thay offer the following as a summary of the key things we need to know for now:
- To create an array or tensor, pass a list (or a list of lists), to *array* or *tensor*:

```python
data = [[1,2,3],[4,5,6]]
arr = array(data)
tns = tensor(data)
```
```python
arr   # numpy
array([[1, 2, 3],
       [4, 5, 6]])
       
tns   # pytorch
tensor([[1, 2, 3],
        [4, 5, 6]])
```
*All the operations that follow are shown on tensors, but the syntax and results for NumPy arrays are identical.*
- Select a row:
```python
tns[1]
```
tensor([4, 5, 6])

- Select a column *(using : to indcate all of the first axis - rows in this case)*:
```python
tns[ ; ,1]
```
tensor([2, 5])

- Combine these with Python slice syntax *([start:end], with end being excluded)* to select part of a row or column:
```python
tns[1, 1:3]
```
tensor([5, 6])

- Use standard operators, such as +, -, *, and /:
```python
tns + 1
```
tensor([[2, 3, 4],
        [5, 6, 7]])
        
- Tensors have a type:
```python
tns.type()
```
'torch.LongTensor'

_ And will automatically change that type as needed; for example, from *int* to *float*:
```python
tns1 = tns * 2.5
tns1
```
tensor([[ 2.5000,  5.0000,  7.5000],
        [10.0000, 12.5000, 15.0000]])
        
```python
tns1.type()
```
'torch.FloatTensor'

## Computing Metrics using Broadcasting

<ins>Metric:</ins> A number calculated based on the predictions of our model and the correct labels, in order to tell us how good our model is.    
In practice, we use ***accuracy*** as the metric for classification models. And we calculate it ***on the validation set***.    
We already have a *validation set* totally separated from the trining data in the *valid* directory.

1. Create tensors for our 3s and 7s from the *valid* directory.
```python
valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape
```
(torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))

We ends up with 2 tensors: One of 1010 images of 28 x 28 pixesl of the number 3 and another of 1028 imgaes of 28 x 28 0f the number 7.

2. We ultimately want to write a function *is_3* that will decide whether an arbitrary image is a 3 or a 7. 
It will do this by deciding which of our two “ideal digits” that arbitrary image is closer to. 
For that we need to define a notion of distance — that is, a function that calculates the distance between two images.
```python
a_3 = stacked_threes[1]
mean3 = stacked_threes.mean(0)
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)
```
tensor(0.1114)

This is the same value we previously calculated for the distance between these two images, the ideal 3 mean_3 and the arbitrary sample 3 a_3, which are both single-image tensors with a shape of [28,28].    
But to calculate a metric for overall accuracy, we will need to calculate the distance to the ideal 3 for every image in the validation set.
We could write a loop over all of the single-image tensors that are stacked within our validation set tensor, valid_3_tens, which has a shape of [1010,28,28] representing 1,010 images. But that would be highly inefficient and there is a better way.    

3. We take the same distance function, designed for comparing two single images, but pass in as an argument *valid_3_tens*, the tensor that represents the 3s validation set, instead of the single image a_3 as we did before:
```python
valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
```
(tensor([0.1290, 0.1223, 0.1380,  ..., 0.1337, 0.1132, 0.1097]),
 torch.Size([1010]))
 
Instead of complaining about shapes not matching, it returned the distance for every single image as a vector (i.e., a rank-1 tensor) of length 1,010 (the number of 3s in our validation set).    
You can see our function mnist_distance has the subtraction (a-b). 
The magic trick is that <ins>PyTorch, when it tries to perform a simple subtraction operation between two tensors of different ranks, 
will use ***broadcasting***: it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank.</ins>.    
After broadcasting so the two argument tensors have the same rank, PyTorch applies its usual logic for two tensors of the same rank: it performs the operation on each corresponding element of the two tensors, and returns the tensor result. For instance:
```python
tensor([1,2,3]) + tensor([1,1,1])
```
tensor([2, 3, 4])

In our case, PyTorch treats mean3, a rank-2 tensor representing a single image, as if it were 1,010 copies of the same image, and then subtracts each of those copies from each 3 in our validation set.
```python
(valid_3_tens-mean3).shape
```
torch.Size([1010, 28, 28])

There are a couple of important points about how broadcasting is implemented, which make it valuable not just for expressivity but also for performance: 
- PyTorch doesn’t actually copy mean3 1,010 times. It pretends it were a tensor of that shape, but doesn’t allocate any additional memory.  
- It does the whole calculation in C (or, if you’re using a GPU, in CUDA, the equivalent of C on the GPU), 
tens of thousands of times faster than pure Python (up to millions of times faster on a GPU!).

> According to the authors, **broadcasting is the most important technique for us to know to create efficient PyTorch code.**

Next in mnist_distance we see *abs* applied to a tensor. It applies the method to each individual element in the tensor, 
and returns a tensor of the results (that is, it applies the method elementwise). 
So in this case, we’ll get back 1,010 matrices of absolute values.

Finally, our function calls mean((-1,-2)). The tuple (-1,-2) represents a range of axes. 
In Python, -1 refers to the last element, and -2 refers to the second-to-last. 
So in this case, this tells PyTorch that we want to take the mean ranging over the values indexed by the last two axes of the tensor. 
The last two axes are the horizontal and vertical dimensions of an image. 
After taking the mean over the last two axes, we are left with just the first tensor axis, which indexes over our images, 
which is why our final size was (1010). In other words, for every image, we averaged the intensity of all the pixels in that image.

4. We can use mnist_distance to figure out whether an image is a 3 by using the following logic:     
if the distance between the digit in question and the ideal 3 is less than the distance to the ideal 7, then it’s a 3.     
This function will automatically do broadcasting and be applied elementwise, just like all PyTorch functions and operators:
```python
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
is_3(a_3), is_3(a_3).float()
```
(tensor(True), tensor(1.))

Thanks to broadcasting, we can also test it on the full validation set of 3s:
```python
is_3(valid_3_tens)
```
tensor([True, True, True,  ..., True, True, True])

5. Now we can calculate the accuracy for each of the 3s and 7s, by taking the average of that function for all 3s and its inverse for all 7s:
```python
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2
```
(tensor(0.9168), tensor(0.9854), tensor(0.9511))

We’re getting over 90% accuracy on both 3s and 7s. But 3s and 7s are very different-looking digits, 
and we’re classifying only 2 out of the 10 possible digits so far. So we’re going to need to do better.    
And to do better, we need to try a system that does some real learning — one that can automatically modify itself to improve its performance.    

it’s time to talk about the **training process and SGD**, but that deerves a separate post.
