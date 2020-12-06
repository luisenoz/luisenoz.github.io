# Putting it all Together

In code, the process to be implemented is going to be something like this for each epoch:

```python
for x,y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    parameters -= parameters.grad * lr
```

1. Reinitialize the parameters:
```python
weights = init_params((28*28,1))
bias = init_params(1)
```
2. Create a Dataloader for a dataset:
```python
dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape
```
(torch.Size([256, 784]), torch.Size([256, 1]))

3. The same for the validation set:
```python
valid_dl = DataLoader(valid_dset, batch_size=256)
```

4. Test with a mini-batch of size 4:
```python
batch = train_x[:4]
batch.shape 
```
torch.Size([4, 784]) 

```python
preds = linear1(batch)
preds 
```
tensor([[-2.1876],
        [-8.3973],
        [ 2.5000],
        [-4.9473]], grad_fn=<AddBackward0>)
        
```python
loss = mnist_loss(preds, train_y[:4])
loss 
```
tensor(0.7419, grad_fn=<MeanBackward0>)

```python
loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad
```
(torch.Size([784, 1]), tensor(-0.0061), tensor([-0.0420])).    

Put all that in a single function:
```python
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
```
And test it:
```python
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad
```
(tensor(-0.0182), tensor([-0.1260]))

But look what happens if we called it twice:
```python
calc_grad(batch, train_y(:4), linnear1)
weights.grad.mean(), bias.grad
```
(tensor(-0.0303), tensor([-0.2100]))

The gradients have changed!     
The reason for this is that *loss.backward* adds the gradients of loss to any gradients that are currently stored.     
So, we have to set the current gradients to 0 first:
```python
weights.grad.zero_()
bias.grad.zero_();
```

> **In Place Operations:** Methods in PyTorch whose names end in an underscore modify their objects in place.     
For instance, *bias.zero_* sets all elements of the tensor bias to 0.

5. The only remaining step is to update the weights and biases based on the gradient and learning rate.     
When we do so, we have to tell PyTorch not to take the gradient of this step too — 
otherwise, things will get confusing when we try to compute the derivative at the next batch!     
To get that, we assign to the *data* attribute of a tensor, so PyTorch will not take the gradient of that step. 
```python
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```
6. Check how we’re doing by looking at the accuracy of the validation set.     
To decide if an output represents a 3 or a 7, we can just check whether it’s greater than 0, (using Broadcasting!).
```python
(preds>0.0).float() == train_y[:4]
```
tensor([[False],
        [False],
        [ True],
        [False]])
        
That gives us this function to calculate our validation accuracy: 
```python
def batch_accuracy(xb, yb):
  preds = xb.sigmoid()
  correct = (preds>0.5) == yb
  return correct.float().mean()
```
```python
batch_accuracy(linear1(batch), train_y[:4])
```
tensor(0.2500).    

Putting the batches together:
```python
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```
```python
validate_epoch(linear1)
```
0.5263

7. Train for 1 epoch and see if the accuracy improves:
```python
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)
```
0.6663

Train a few more:
```python
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
```
0.8265 0.8899 0.9182 0.9275 0.9397 0.9466 0.9505 0.9525 0.9559 0.9578 
0.9598 0.9608 0.9613 0.9618 0.9632 0.9637 0.9647 0.9657 0.9672 0.9677 

It seems to be working!We’ve created a general-purpose foundation we can build on.    
Our next step will be to create an object that will handle the SGD step for us. That object in PyTorch is called an *optimizer*.

## Creating an Optimizer

1. Replace our *linear1* function by PyTorch *nn.Linear* module.     
A *module* is an object of a Class that inherits from the PyTorch *nn.Module* Class.     
As in Python functions, you can call objects of this class using parenthesis, and they will return the activation of a model.     
*nn.Linear* does the same thing as our *init_params* and *linear* together. It contains both the weights and biases in a single class.

```python
linear_model = nn.Linear(28*28,1)
```
Every PyTorch module knows what parameters it has that can be trained; they are available through the parameters method:
```python
w,b = linear_model.parameters()
w.shape,b.shape
```
(torch.Size([1, 784]), torch.Size([1]))

2. Create the *optimizer*:
```python
class BasicOptim:
    def __init__(self,params,lr): 
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```
We can create our optimizer by passing in the model’s parameters:
```python
opt = BasicOptim(linear_model.parameters(), lr)
```
3. And then simplfy the training loop:
```python
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```
4. And the validation fucntion doesn't need to change:
```python
validate_epoch(linear_model)
```
0,5516

5. We can aslo simplify the training loop:
```python
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
```
And The results should be the same as in the previous section:     
```python
train_model(linear_model, 20)
```
0.4932 0.79 0.8559 0.9174 0.936 0.9502 0.957 0.9629 0.9658 0.9687 
0.9702 0.9716 0.9741 0.9746 0.976 0.976 0.9775 0.9775 0.9785 0.9785

```python
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
```
0.4932 0.9262 0.79 0.9048 0.9296 0.9448 0.955 0.9619 0.9648 0.9663 
0.9692 0.9702 0.9741 0.9751 0.9755 0.9765 0.977 0.978 0.978 0.9785 

6. fastai also provides Learner.fit, which we can use instead of train_model.     
To create a Learner, we first need to create a DataLoaders, by passing in our training and validation DataLoaders:
```python
dls = DataLoaders(dl, valid_dl)
```

7. To create a Learner without using an application (such as cnn_learner), 
we need to pass in all the elements that we’ve created in this chapter:     
the DataLoaders, the model, the optimization function (which will be passed the parameters), 
the loss function, and optionally any metrics to print:
```python
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
```
8. And finally call *fit*:
```python
learn.fit(10, lr=lr)
```
|epoch	|train_loss	|valid_loss	|batch_accuracy	|time  |   
|:---|:---|:---|:---|:---|
|0	    |0.637393	  |0.502965	  |0.495584	      |00:00 |
|1	    |0.402730	  |0.271369	  |0.745339	      |00:00 |
|2	    |0.152630	  |0.163603	  |0.853778	      |00:00 |
|3	    |0.069656	  |0.100934	  |0.917076	      |00:00 |
|4	    |0.038907	  |0.074887	  |0.934249	      |00:00 |
|5	    |0.026727	  |0.060520	  |0.949460	      |00:00 |
|6	    |0.021618	  |0.051534	  |0.955839 	    |00:00 |
|7	    |0.019278	  |0.045524	  |0.962709	      |00:00 |
|8	    |0.018049	  |0.041267	  |0.965653	      |00:00 |
|9	    |0.017285	  |0.038108	  |0.968106	      |00:00 |

With these classes, we can now replace our linear model with a neural network.



