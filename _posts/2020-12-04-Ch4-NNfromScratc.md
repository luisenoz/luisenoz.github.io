# Getting to our first Neural Network from scratch

It was kind of magical to follow the steps Jeremy put in the book to lead us in a way from the most elemental steps 
to reach, almost without noticing until the end, the creation of a simple NN.

## The MNIST Loss Function

We’ll need to concatenate *(using torch.cat)* our "xs" (the images) into a single tensor, and also change them from a list of matrices (a rank-3 tensor)
to a list of vectors (a rank-2 tensor).     
We can do this using *view*, a PyTorch method that changes the shape of a tensor without changing its contents. 
And we can use *-1* as a special parameter to *view* that will make the axis as big as necessary to fit all the data:
```python
train_x = torch.cat([stacked_threes, 
stacked_sevens]).view(-1, 28*28)
```
Then, we'll need to create our "ys" (the labels). We'll use 1 for 3s and 0 for 7s. 
We use *unsqueeze*, that is a PyTorch method that returns a new tensor with a dimension of size one inserted at the specified position (in our case at index 1):
```python
train_y = tensor([1]*len(threes) + 
[0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape
```
(torch.Size([12396, 784]), torch.Size([12396, 1])).    
Now we have our "xs" in a tensor of 12396 rows and 784 columns (=28x28) and our "ys" in one of 12396 for 1 column.

A Dataset in PyTorch is required to return a tuple of (x,y) when indexed.     
We can use *zip* combined with *list* to get it done:
```python
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y
```
(torch.Size([784]), tensor([1]))

We can summarise all the previous steps as follows:
```python
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

We can define a function to calculate random weights for every pixel:

```python
def init_params(size, std=1.0): 
  return (torch.randn(size)*std).requires_grad_()
```
```python
weights = init_params((28*28,1))
```

But, the function *weights*pixels* will be always equal to 0 when the pixels are equal to 0.     
We need to add the "b" from the formula for a line (y=w*x+b). We’ll initialize it to a random number too:
```python
bias = init_params(1)
```


> In neural networks jargon => **ws = weights**, **b = bias**; and both are **parameters**

We can now calculate the prediction for 1 image:
```pyton
(train_x[0]*weights.T).sum() + bias
```
tensor([20.2336], grad_fn=<AddBackward0>)

Using *matrix multiplication* we can calculate *w*x* for every row of a matrix.     
In Python, matrix multiplication is represented with the **@** operator.
```python
def linear1(xb): 
  return xb@weights + bias
preds = linear1(train_x)
preds
```
tensor([[20.2336],
        [17.0644],
        [15.2384],
        ...,
        [18.3804],
        [23.8567],
        [28.6816]], grad_fn=<AddBackward0>)

The first element is the same as we calculated before, as we’d expect.     
This equation, **batch @ weights + bias**, is one of the two fundamental equations of any neural network, (the other is the *activation function*).

For checking accuracy, we can decide if an output represents a 3 or a 7 by just checking whether it’s greater than 0, 
so our accuracy for each item can be calculated using broadcasting as follows:
```python
corrects = (preds>0.0).float() == train_y
corrects 
```
tensor([[ True],
        [ True],
        [ True],
        ...,
        [False],
        [False],
        [False]])
```python
corrects.float().mean().item()
```
0.4912068545818329

Let’s see what the change in accuracy is for a small change in one of the weights:
```python
weights[0] *= 1.0001 

preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()
```
0.4912068545818329

- We need gradients in order to improve our model using SGD.
- And in order to calculate gradients we need a loss function that represents how good our model is. 
- That is because the gradients are a measure of how that loss function changes with small tweaks to the weights.
- The obvious approach would be to use accuracy, which is our metric, as our loss function as well.
- Unfortunately, we have a significant technical problem to use accuracy as the loss fucntion.
- The gradient of a function is its slope, which can be defined as rise over run — 
that is, how much the value of the function goes up or down, divided by how much we changed the input.
     (y_new – y_old) / (x_new – x_old)
- The problem is that a small change in weights from *x_old* to *x_new* isn’t likely to cause any prediction to change, 
so (y_new – y_old) will almost always be 0. In other words, the gradient is 0 almost everywhere.
- A very small change in the value of a weight will often not change the accuracy at all. 
This means it is not useful to use accuracy as a loss function

> In mathematical terms, accuracy is a function that is constant almost everywhere (except at the threshold, 0.5), 
so its derivative is nil almost everywhere (and infinity at the threshold).     
This then gives gradients that are 0 or infinite, which are useless for updating the model.

- We need a loss function that, when our weights result in slightly better predictions, gives us a slightly better loss.

We can defina a function for the new loss function:     
1. One argument, *prds*, of values between 0 and 1, where each value is the prediction that an image is a 3. 
It is a vector (i.e., a rank-1 tensor) indexed over the images.     

2. Another argument, *trgts*, also with values of 0 or 1 that tells whether an image actually is a 3 or not. 
It is also a vector (i.e., another rank-1 tensor) indexed over the images.

*As an example, let's assume our model predicted with high confidence (0.9) that the first was a 3, with slight confidence (0.4)that the second was a 7, 
and with fair confidence (0.2), but incorrectly, that the last was a 7.*

```python
prds = tensor([0.9, 0.4, 0.2)]
```
```python
trgts = tensor(1, 0, 1)
```
3. We can define a new loss function to meassure the differences between predictions and targets:     
```python
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 
    1-predictions, predictions).mean()
```
The function *torch.where(a,b,c)* is the same as running the list comprehension:     
[b[i] if a[i] else c[i] for i in range(len(a))],     
except it works on tensors, at C/CUDA speed.     
In plain English, this function will measure how distant each prediction is from 1 if it should be 1, 
and how distant it is from 0 if it should be 0, and then it will take the mean of all those distances.

Applied to our exampple:    
```python
torch.where(trgts==1, 1-prds, prds)
```
tensor([0.1000, 0.4000, 0.8000])

This function returns:     
- a lower number when predictions are more accurate,
- when accurate predictions are more confident (higher absolute values),
- and when inaccurate predictions are less confident.     
It's coherent with our assumption that a lower value of a loss function is better.

4. Since we need a scalar for the final loss, *mnist_loss* takes the mean of the previous tensor:
```python
mnist_loss(prds,trgts)
```
tensor(0.4333)

5. To test the new function, if we change our prediction for the one “false” target from 0.2 to 0.8, 
the loss should go down, indicating that this is a better prediction:     
```python
mnist_loss(tensor([0.9, 0.4, 0.8]),trgts)
```
tensor(0.2333) *(from 0.4333 before)*

One problem with *mnist_loss* as currently defined is that it assumes that predictions are always between 0 and 1.     

### SIGMOID

The sigmoid function always outputs a number between 0 and 1.     
It’s defined as follows: 
```python
def sigmoid(x): 
  return 1/(1+torch.exp(-x))
```
But PyTorch defines an accelerated version for us, so we don’t really need our own.




  
