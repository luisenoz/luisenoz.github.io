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

## Working in the Notebook:

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
4. With the key saved, we can start using the function *search_images_bing*. This function is provided within utils class included with the notebooks online.  
*At any time, if you’re not sure where a function is defined, you can just type it in the notebook to find out:*

```python
search_images_bing
<function fastbook.search_images_bing(key, term, min_sz=128)>
```

5. Now, using the *searc_image_bing* function, Ill try to download the URLs of Great Barrier Reef images, or wahtever Bing Image Search finds for that search term:
```python
results = search_images_bing(key, 'great barrier reef')
ims = results.attrgot('content_url')
len(ims)
```

And needless to say, the result was: **ErrorResponseException: Operation returned an invalid status code 'PermissionDenied'**.  

I spent half a day searching for a solution, tried a few but haven't had any success. I need a break now and hope to find something to break this big roadblock tomorrow.
Wel, what. difference makes to stsrt afresh!  
The problem was that the original *searh_image-bing* function developed by Jeremy and included in the book doesn't work anymore, due to the chnages that Microsost inserted to the links in Bing Search.
Fortunately, just yesterday the user "retuso" put a redefined *searh_image-bing* function for all of us to use in the forum, and that function worked beautifully!!!
Many thanks to all people colaborating in the forum, but specially to "retuso" in this particular ocassion.  
Just in case, the whole process to make it work is as follows:
- In a cell after the saved the key, insert the new redefined function from "retuso":
```python
def search_images_bing(key, term, max_images: int = 100, **kwargs):    
     params = {'q':term, 'count':max_images}
     headers = {"Ocp-Apim-Subscription-Key":key}
     search_url = "https://api.bing.microsoft.com/v7.0/images/search"
     response = requests.get(search_url, headers=headers, params=params)
     response.raise_for_status()
     search_results = response.json()    
     return L(search_results['value'])
     ```
 - In the following cell, maintain the same original code, but change ('content_url') for ('contentUrl') as here:
 ```python
 results = search_images_bing(key, 'great barrier reef')
ims = results.attrgot('contentUrl')
len(ims)```

- You should get a number 100 when running it. That means we successfully downloaded the URLs of 100 images that Bing Image Search found for our search term.

6. At the moment, I have a variable, *ims* that contains the URLS of 100 imagens Bing Search relates to "great barrier reef". But what if I wanted to access and see the photos?  
a. We need to have a folder created where we can download the images.
  a1. Verify current directory: os.getcwd()
  a2. You can use the original *images* directory, or create a new one for the new images: os.mkdir("myimages")
  a3. We can check files and directories in our current position by running: os.listdir()
  a4. Or we can find the location of a particular file or directory using: os.path.abspath("myimages")  
b. We need a function that would download the photos (I started with only one, as a test) to the indicated folder:
```python
dest = 'myimages/gbr.jpg'
download_url(ims[0], dest)
```
The first line indicates the path to save the photo and the name and format of the file.
The second uses what i believe is a fastai function *download_url* to precisely get the image in th first url in our ims variable and save into the predefined directory. 
c. And another function to access the photo:
```python
im = Image.open(dest)
im.to_thumb(128,128)
```
**I'm not even going to try to explain the feeling when the tiny thumb with an image of the Great Baarrier Reef showed up on my notebook after running that cell!!!**



