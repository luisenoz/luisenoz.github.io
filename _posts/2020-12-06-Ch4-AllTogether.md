# CHAPTER 4.6: PUTTING IT ALL TOGETHER

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
|6	    |0.021618	  |0.051534	  |0.955839 	  |00:00 |
|7	    |0.019278	  |0.045524	  |0.962709	      |00:00 |
|8	    |0.018049	  |0.041267	  |0.965653	      |00:00 |
|9	    |0.017285	  |0.038108	  |0.968106	      |00:00 |

With these classes, we can now replace our linear model with a neural network.

## Adding a Nonlinearity

So far, we tried optimising the parameters of a a simple linear function. But a linear classifier is constrained in terms of what it can achieve.    
To make it more complex, and able to handle more and complex tasks, we need to add some nonlinearity between two linear classifiers; and that will give us a neural network.

The following encapsulates the entire definition of a basic neural network:
```python
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```
All we have in *simple_net* is two linear classifiers with a *max* function between them.     
*w1 and w2* are weight tensors, and *b1 and b2* are bias tensors. They are the parameters that are initially randomly initialized, 
just as we did in the previous section:
```python
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```
The key point is that *w1* has 30 output activations (which means that *w2* must have 30 input activations, so they match).     
That means that the first layer can construct 30 different features, each representing a different mix of pixels, *(and we can change that 30 to anything we like, to make the model more or less complex)*.     
That little function *res.max(tensor(0.0))* is called a *rectified linear unit*, also known as ***ReLU***.     
It sounds complicated but it just replaces every negative number with a 0.  
It is also available in PyTorch as *F.relu*:
```python
plot_function(F.relu)
```
![relu](https://github.com/luisenoz/luisenoz.github.io/blob/master/images/relu.png)

The basic idea is that by using more linear layers, we can have our model do more computation, 
and therefore model more complex functions.     
But there’s no point in just putting one linear layout directly after another one, 
because a series of any number of linear layers in a row can be replaced with a single linear layer with a different set of parameters.     
But if we put a nonlinear function between them, such as max, this is no longer true. 
Now each linear layer is somewhat decoupled from the other ones and can do its own useful work.

Amazingly, it can be mathematically proven that this little function can solve any computable problem to an arbitrarily high level of accuracy, if we can find the right parameters for w1 and w2 and if we make these matrices big enough.     
> Mathematically speaking, any neural network architecture aims at finding any mathematical function y= f(x) that can map attributes(x) to output(y). 
The accuracy of this function i.e. mapping differs depending on the distribution of the dataset and the architecture of the network employed. 
The function f(x) can be arbitrarily complex. The **Univeral Approximation Theorem** tells us that Neural Networks has a kind of universality 
i.e. no matter what f(x) is, there is a network that can approximately approach the result and do the job! This result holds for any number of inputs and outputs.

The three lines of code *(res=)* that we have in the fucntion are known as *layers*:
- The first and third are known as *linear layers*, 
- and the second line of code is known variously as a nonlinearity, or *activation function*.

As usual, taking advantage of Pytorch, we can replace that code for something evan simpler:
```python
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```

- *nn.Sequential* creates a module that will call each of the listed layers or functions in turn.
- *nn.ReLU* is a PyTorch module that does exactly the same thing as the *F.relu* function.     
*(Most functions that can appear in a model also have identical forms that are modules, generally, by just replacing F with nn and changing the capitalization)*.    
When using nn.Sequential, PyTorch requires us to use the module version. 
Since modules are classes, we have to instantiate them, which is why we used nn.ReLU.

Because nn.Sequential is a module, we can get its parameters, which will return a list of all the parameters of all the modules it contains:
```python
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)
                
learn.fit(40, 0.1)
```
|epoch	|train_loss	|valid_loss	|batch_accuracy	|time  |   
|:---|:---|:---|:---|:---|
|0	|0.333021	|0.396112	|0.512267	|00:00|
|1	|0.152461	|0.235238	|0.797350	|00:00|
|2	|0.083573	|0.117471	|0.911678	|00:00|
|3	|0.054309	|0.078720	|0.940628	|00:00|
|4	|0.040829	|0.061228	|0.956330	|00:00|
|..	|........	|........	|........	|.....|
|..	|........	|........	|........	|.....|
|35	|0.014686	|0.021184	|0.982336	|00:00|
|36	|0.014549	|0.021019	|0.982336   |00:00|
|37	|0.014417	|0.020864	|0.982336	|00:00|
|38	|0.014290	|0.020716	|0.982336	|00:00|
|39	|0.014168	|0.020576	|0.982336	|00:00|

The training process is recorded in *learn.recorder*, with the table of output stored in the values attribute, 
so we can plot the accuracy over training:
```python
plt.plot(L(learn.recorder.values).itemgot(2));
```
And we can get the final accuracy:
```python
learn.recorder.values[-1][2]
```
0.982336

At this point, we have reached something that is rather magical:
- A function that can solve any problem to any level of accuracy (the neural network) given the correct set of parameters.
- A way to find the best set of parameters for any function (stochastic gradient descent).

### Going deeper

There is no need to stop at just two linear layers. We can add as many as we want, as long as we add a nonlinearity between each pair of linear layers.     
However, the deeper the model gets, the harder it is to optimize the parameters in practice.     
If we already know that a single nonlinearity with two linear layers is enough to approximate any function, why would we use deeper models?     
The reason is performance. With a deeper model (one with more layers), we do not need to use as many parameters; we can use smaller matrices, 
with more layers, and get better results than we would get with larger matrices and few layers.     
That means that we can train the model more quickly, and it will take up less memory.

The book shows what happens when we train an 18-layer model using the same approach we saw in Chapter 1:
```python
dls = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```
|epoch	|train_loss	|valid_loss	|batch_accuracy	|time  |   
|:---|:---|:---|:---|:---|
|0	|0.078008	|0.024554	|0.995584	|00:15|

Nearly 100% accuracy in less than 15'!

We already have the foundational pieces, but there are just a few little tricks we need to use to get such great results from scratch ourself. 
*(Of course, even if we know all the tricks, we’ll nearly always want to work with the prebuilt classes provided by PyTorch and fastai, 
because they'll save us from having to think about all the little details ourself.)*

## Jargon:

A neural network contains a lot of numbers, but they are only of two types: 
1. numbers that are calculated, 
2. and the parameters that these numbers are calculated from.     
This gives us the two most important pieces of jargon to learn:

***Activations***.    
Numbers that are calculated (both by linear and nonlinear layers).    

***Parameters***.    
Numbers that are randomly initialized, and optimized (that is, the numbers that define the model)

Our activations and parameters are all contained in ***tensors***.     
These are simply regularly shaped arrays.     
The number of dimensions of a tensor is its *rank*. There are some special tensors:
- Rank-0: scalar  
- Rank-1: vector  
- Rank-2: matrix

A neural network contains a number of ***layers***. Each layer is either ***linear or nonlinear***.     
Sometimes a nonlinearity is referred to as an ***activation function***.

***ReLu***: A type of activation function that returns 0 for negative numbers and doesn't change the positive ones.

***Mini-Batch***: A small group of inputs and labels gathered together in two arrays. 
A gradient descent step is updated in this batch (rather than a whole epoch).

***Forward pass***: Applying a model to some inputs and computing the predictions.

***Loss***: A value that represents how well or badly the model is performing.

***Gradient***: The derivative of the loss with respect to some parameter of the model.

***Backward pass***: Computing the gradient of the loss for all the parameters of the model.

***Gradient Descent***: Taking a step on the opositte direction of the gradient to gradually improve the model's parameters.

***Learning rate***: The size of the step we take when doing SGD to update the model's parameters.


