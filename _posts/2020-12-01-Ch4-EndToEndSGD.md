# An End to End SGD Example

The book starts the example with a simple, synthetic example model, where we imagine we were measuring the speed of a roller coaster.     
If we were measuring the speed manually every second for 20 seconds, it might look something like this:
```python
time = torch.arange(0,20).float(); time
```
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,14., 15., 16., 17., 18., 19.]).    
```python
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);
```
*(The authors added a bit of random noise, since measuring things manually isn’t precise)*

Using SGD, we can try to find a function that matches our observations.     
We can’t consider every possible function, so let’s use a guess that it will be quadratic; i.e., a function of the form *a*(time**2)+(b*time)+c*.

To distinguish clearly between the function’s input (the time when we are measuring the coaster’s speed) 
and its parameters (the values that define which quadratic we’re trying); we can collect the parameters in one argument 
and thus separate the input, *t*, and the parameters, *params*, in the function’s signature:

```python
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c
```
We restricted the problem of finding the best imaginable function that fits the data to finding the best *quadratic* function.     
And now, to find the best quadratic function, we need to find only the best values for a, b, and c.

We need to define first what we mean by “best”, by choosing a *loss function*.     
That loss function will return a value based on a prediction and a target, where lower values of the function correspond to “better” predictions.     
For continuous data, it’s common to use *mean squared error*:
```python
def mse(preds, targets): return ((preds-targets)**2).mean()
```

Now we can go through the mentioned seven-step process:

## Step 1: Initialise the parameters
We initialize the parameters to random values and tell PyTorch that we want to track their gradients using *requires_grad_*:
```python
params = torch.randn(3).requires_grad_()
```
And we can clone the original parameters into a new variable to have them available later on:
```python
orig_params = params.clone()
```

## Step 2: Calculate the Predictions
```python
preds = f(time, params)
```
Now we can create a little function to see how close our predictions are to our targets, and take a look:
```python
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)
```
```python
show_preds(preds)
```

## Step 3: Calculate the loss
```python
loss = mse(preds, speed)
loss
```
tensor(25823.8086, grad_fn=<MeanBackward0>).    
Since our goal is now to improve this, we’ll need to know the gradients.

## Step 4: Calculate the gradients
```python
loss.backward()
params.grad
```
tensor([-53195.8633,  -3419.7148,   -253.8908]).    

```python
params.grad * 1e-5
```
tensor([-0.5320, -0.0342, -0.0025]).    
We can use these gradients to improve our parameters.     
We’ll need to pick a learning rate, and for now we’ll just use 1e-5 or 0.00001:

```python
params
```
tensor([-0.7658, -0.7506,  1.3525], requires_grad=True)

## Step 5: Step the weights
We need to update the parameters based on the gradients we just calculated:
```python
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None
```
> A very useful clarification FROM ALEXIS:      
Understanding this bit depends on remembering recent history:     
To calculate the gradients, we **call backward on the loss**.     
But this loss was **itself calculated by mse**,     
which in turn **took preds as an input**,     
which was **calculated using f taking as an input params**,     
which was **the object on which we originally called required_grads_**     
which is **the original call that now allows us to call backward on loss**.     
This chain of function calls represents the mathematical composition of functions, 
which enables PyTorch to use calculus’s chain rule under the hood to calculate these gradients.

To see if the loss has improved with the new parameters:
```python
preds = f(time,params)
mse(preds, speed)
```
tensor(2470.4714, grad_fn=<MeanBackward0>)

We need to repeat this a few times, so we’ll create a function to apply one step:
```python
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds
```

## Step 6: Repeat the process
By looping 10 times in this case, and performing many improvements, we hope to reach a good result:
```python
for i in range(10): apply_step(params)
```
2470.471435546875.    
1010.0901489257812.    
733.7384033203125.    
681.4413452148438.    
671.5423583984375.    
669.666259765625.    
669.3082885742188.    
669.2379150390625.    
669.2215576171875.    
669.2156372070312. 

The loss is going down, just as expected. However, looking only at these loss numbers 
disguises the fact that each iteration represents an entirely different quadratic function being tried, 
on the way to finding the best possible quadratic function.     
We can see this process visually if, instead of printing out the loss function, we plot the function at every step. 
Then we can see how the shape is approaching the best possible quadratic function for our data:

```python
_,axs = plt.subplots(1,4,figsize=(12,3))
for ax in axs: show_preds(apply_step(params, False), ax)
plt.tight_layout()
```

![FunctionApproachBest](https://github.com/luisenoz/luisenoz.github.io/blob/master/images/QuadrCurveShapes.png)

## Step 7: Stop
For the example, we stopped after 10 epochs, but in practice, we would watch the training and validation losses and our metrics to decide when to stop.

# Summarizing Gradient Descent


