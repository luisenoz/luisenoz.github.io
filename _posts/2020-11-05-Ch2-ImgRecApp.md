# Creating my Image Recognition App!

The project introduced in the book in this chapter is a *bear detector* that discriminates between three types of bear: grizzly, black, and teddy bears.  
However, and following Jeremy's advise, I'll follow the instructions in the book but based on my own set of images. I'm sure I'll find more than a few obstacles,
but I don't know of a beter way to learn something than trying to do it!

At the time of writing the book, Bing Image Search was the best option for finding and downloading images.  
To download images with Bing Image Search, you need to sign up at Microsoft for a free account and you will be given a key 
that you'll need to include in the code when downloading images.
Unfortunately, I started this blog after getting my key and now I cannot reconstruct the path I followed; which would have been helpful, 
becasue the process to get that key is not what I'd describe as an straight forward one.  
However, I'm sure if you follow the instructions in the book and some help from Google, you should be able to eventually get it.

## Defining the images

If possible, I'd love to develop an app that can identify some of the most famous places in Australia.  
So, although it is difficult, at least to me, to reduce Australia to just a bunch of sites, I'll try to do experiment by defining only 5 places to start with:
- Great Barrier Reef
- Sydney Opera House
- Ayers rock/Uluru
- The Twelve Apostles
- Wave Rock

**Disclaimer**: If you're Australian or have been here, some of my selections are obvious (*The Great Barrier Reef or the Opera House*) 
while others could arguably be replaced by some places more popular or representative; but my intention is to develop and app that can identify some places in Australia,
and the five selected are good enough for the objective. I'm not tryng to identify the top 5 place in OZ!

## Working in the Notebbok:

1. Created a new notebook (*In a shock of orgininality I named it "My Image Recognition App"*). But it wasn't the start I was waiting for!  

2. I copied the first 2 cells from the orginal *bear recognition* notebook. Supposedly, running those cells would install/upgrade the fastbook library and others, 
but instead I'v got the following error when tried to run it:

> ERROR: After October 2020 you may experience errors when installing or updating packages. This is because pip will change the way that it resolves dependency conflicts.
We recommend you use --use-feature=2020-resolver to test your packages with the new resolver before it becomes the default.
voila 0.2.4 requires nbconvert<7,>=6.0.0, but you'll have nbconvert 5.6.1 which is incompatible.
fastai 2.1.4 requires fastcore>=1.3.0, but you'll have fastcore 1.1.2 which is incompatible.

From a look in Stackoverflow (where else!), I found the follwing explanation:  
*"pip will introduce a new dependency resolver in October 2020, which will be more robust but might break some existing setups.
Therefore they are suggesting users to try running their pip install scripts at least once (in dev mode) with this option: 
--use-feature=2020-resolver to anticipate any potential issue before the new resolver becomes the default in October 2020 with pip version 20.3."*. 

- Therefore, since the original code was:  
```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```
  
- I tried:  
```python
!pip install -Uqq fastbook--use-feature=2020-resolver
import fastbook
fastbook.setup_book()
```
- But I'v got the following:
> ERROR: Invalid requirement: 'fastbook--use-feature=2020-resolver'
Hint: = is not a valid operator. Did you mean == ?.

- However, (*and don't ask me why?*), I run the original cell without the resolver again, and it worked!!!

3. I saved my key from Microsoft Azure in a variable so I can use it when downloading images from Bing Image Search:
```python
key = os.environ.get('AZURE_SEARCH_KEY', '##############################')
```











