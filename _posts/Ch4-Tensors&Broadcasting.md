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

