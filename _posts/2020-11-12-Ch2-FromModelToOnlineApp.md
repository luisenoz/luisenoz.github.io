# CHAPTER 2.4: FROM A TRAINED MODEL TO AN ONLINE APPLICATION!!!

Once we’ve got a model ready, we need to save it so we can then copy it over to a server where we can use it in production.  
A model consists of two parts: 
- the architecture 
- and the trained parameters.  
We need to save both of these, because then, when we load the model, we'll be sure that we have the matching architecture and parameters.  
To save both parts, we use the *export* method.
```python
learn.export()
```
This method even saves the definition of how to create your DataLoaders.  
This is important, because otherwise you would have to redefine how to transform your data in order to use your model in production.  
*(fastai automatically uses your validation set DataLoader for inference by default, 
so your data augmentation will not be applied, which is generally what you want)*

When you call *export*, fastai will save a file called ***export.pkl***
We can check that the file exists, by using the *ls* method that fastai adds to Python’s *Path* class:
```python
path = Path()
path.ls(file_exts='.pkl')
```
(#1) [Path('export.pkl')]

```python
os.path.abspath("export.pkl")
```
'/storage/mynotebooks/export.pkl'. 

We’ll need this file wherever we decide to deploy our app to.

When we use a model for getting predictions, instead of *training*, it's called ***inference***.  
To create our inference learner from the exported file, we use *load_learner*.  
```python
learn_inf = load_learner(path/'export.pkl')
```

When we’re doing inference, we’re generally getting predictions for just one image at a time, passing just a filename to predict:
```python
learn_inf.predict('myimages/gbr.jpg')
```
('Great Barrier Reef',
 TensorImage(1),
 TensorImage([1.2173e-04, 7.9266e-01, 4.5683e-04, 2.0556e-01, 1.1961e-03]))
 
 It returns 3 things: 
 1. the predicted category in the same format we originally provided,
 2. the index of the predicted category, 
 3. and the probabilities of each category.  
*(The last two are based on the order of categories in the *vocab* of the DataLoaders)*
```python
learn_inf.dls.vocab
```
['Ayers Rock', 'Great Barrier Reef', 'Sydney Opera House', 'The Twelve Apostles', 'Wave Rock']

## Creating a Notebook App for our model

To use our model in an application, we can simply treat the predict method as a regular function.  
Therefore, creating an app from the model can be done using any of the many frameworks and techniques available to application developers.  
However, there is no need (at least during our first steps) to get familiar with the world of web application development.  
**We can create a complete working web application using nothing but Jupyter notebooks!**. 
The two things we need for that are as follows: 
- IPython widgets (ipywidgets)  
- Voilà

IPython widgets are GUI components that bring together JavaScript and Python functionality in a web browser, and can be created and used within a Jupyter notebook. 
However, we don’t want to require users of our application to run Jupyter themselves.  
That is why Voilà exists. It is a system for making applications consisting of IPython widgets available to end users, 
without them having to use Jupyter at all. Essentially, it helps us automatically convert the complex web application we’ve already implicitly made (the notebook) 
into a simpler, easier-to-deploy web application, which functions like a normal web application rather than like a notebook.

We can build up our GUI step by step to create a simple image classifier.
1. We need a file upload widget:
```python
btn_upload = widgets.FileUpload()
btn_upload
```
2. Click the Upload botton in the notebook and select a picture.
3. Grab the image:
```python
img = PILImage.create(btn_upload.data[-1])
```
4. Use an *Output* widget to display it:
```python
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl
```
5. Make predictions:
```python
pred,pred_idx,probs = learn_inf.predict(img)
```
6. use a *Label* to display them:
```python
lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred
```
7. Create a a button to do the classification:
```python
btn_run = widgets.Button(description='Classify')
btn_run
```
8. Create a *click event handler*; that is, a function that will be called when it’s pressed, (we define this function just by copu=ying the previous lines of code):
```python
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
btn_run.on_click(on_click_classify)
```
9. If we click the button now, we should see the image and predictions update automatically!  
10. Put back *btn_upload* to a widget for next cell
```python
btn_upload = widgets.FileUpload()
```
11. Put them all in a vertical box (*VBox*) to complete our GUI:
```python
VBox([widgets.Label('Select your place in Australia!'), 
      btn_upload, btn_run, out_pl, lbl_pred])
```

We have written all the code necessary for our app. The next step is to convert it into something we can deploy.

## Turning the Notebook into a real APP

We'll create a new notebook and copy to it only the code needed to create and show the widgets that you need, 
and the Markdown cells for any text that you want to appear.
The books offers a notebook called "bear classifier" that we can copy and update to our set of data, with minimun changes.

Next, we need to install Voilà by copying these lines into a notebook cell and executing it:
```python
!pip install voila
!jupyter serverextension enable voila --sys-prefix
```
The first line installs the voila library and application, and the second connects it to your existing Jupyter notebook.

Voilà runs Jupyter notebooks just like the Jupyter notebook server we you are using, but it removes all of the cell inputs, 
and shows only output (including ipywidgets), along with any Markdown cells. <ins>So what’s left is a web application!</ins>. 

The book says that to view the notebook as a Voilà web application, we need to replace the word “notebooks” in our browser’s URL with “voila/render”.   
However, I initially got an *404 : Not Found - You are requesting a page that does not exist!* message, instead of a Voila web app.  
I tried again by reinstalling Voila, and now I've got *inconsistent versions of Voila, nbdev and nbconvert packages*.

Despite the error message, I have Voila installed (there is a Voila icon in the top menu), but when I try to run the notebook in Voila, I receive an error message:
*ERROR: Voila 0.2.4 requires nbconvert<7, >=6, but you have nbconvert 6.0.7 which is incompatible.*. 
Therefore, I’m in a situation where Voila requires nbconvert >=6 while at the same time nbdev requires the same nbconvert to be <6!

I spent a week stuck with this issue and couldn't find a soltion in the forum or the web. I opened a new topic in fastai forum and I hope somebody would be able to help me.

> **I'd recomend not to continue with the next steps (deploying your model in Bing) until you tested that you can open you model as a Voila web application from the notebook first.**   
**I've tried multiple times, looked on the different libraires instructions and created a topice about my probelm in fastai forum, without any success.**    
*I know it can sound a bit arrogant or selfish, but I felt the forum and the authors let me down with this issue. Not because they didn't provide a solution but becasue nodody replied anything to my topic after more than 2 weeks. I even added  second post asking for at least some references to documenation or other sites that could help me, and again no response. Sorry for venting here, but if I can't not do it in my blog, where could I do it!!!)*

Once you have Voila workiing in your notebook, you have to export your learning model to export.pkl file and download that file, along with the minimal version of the notebook and then push all that plus the requirement.txt file into a new repository in Github. Since it's a bit more complicated than what I thought, 
I'll copy here a detailed process provided in the forum by member vikbehal:

- Build the model and app notebook:
1. Once the model is trained and the notebook is ready, create a duplicate of your working notebook but with minimal content. *(as I mentioned at the beginning of this topic, the books offers a notebook called "bear classifier" that we can copy and update to our set of data, with minimun changes)*.
2. Once done, download the export.pkl file & the minimal version of your notebook created in step 1. *(In my case, I'm working with Paperspace Gradient but downloaded the files to a drive in my local machine)*.  
*(<ins>Note:</ins>To download, just select the files from Github where you have them and click Download at the top of the list.)*

- Setup repository in Github:
1. Create a Github account if you don't have one.
2. Create a new repository in Github.  
<ins>Note:</ins> You can choose whatever name you like as Repository Name, the Type should be ‘Public’ and you can check ‘Initialize this repository with a README’ option. Finally, click on Create repository button to create it.
3. Upload your model (the export.pkl file), the notebook and the requirements.txt file to this repository.  
 **<ins>Note:</ins> Github won't upload any file lager than 25MBs, therefore, if your *export.pkl* is larger than 25 MBs (which is most likely true), you will need to use Git Large File Storage service to be able to upload it. You can do that by following the steps described here:**. 
 
- You need to have Git installed locally in your machine.
- Download and install *Git Large File Storage* [GLF](https://git-lfs.github.com/)
- Next, create a new folder to hold your Github repository in one of your drives.
- On your Terminal head over to the above-created folder by using “cd” command.
- Next, clone your Github repository using “git clone (the URL of your repo)”.  
*(in my case: git clone https://github.com/#######/Aus-Places-Classifier.git). 
(you can get the url from the green CODE button at the top of your Github repo)*.  
<ins>Note:</ins> You may be prompted for username and password.

- Finally, run the below commands in same order to upload the large file.  
Run it in the folder of the cloned Github repository that was created in your machine in the last point.  
```
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add export.pkl
git commit -m "Add model"
git push -u origin main
```
- This should have uploaded your large model file to the Github repo. *.


## Deploying the app

We do not need a GPU to serve our model in production.
GPUs are useful only when they do lots of identical work in parallel. But you’ll normally be classifying just one user’s image at a time. So, a CPU will often be more cost-effective.  
Because of the complexity of GPU serving, many systems have sprung up to try to automate this. However, managing and running these systems is also complex, and generally requires compiling your model into a different form that’s specialized for that system. It’s typically preferable to avoid dealing with this complexity until/unless your app gets popular enough that it makes clear financial sense for you to do so.  
For the initial prototype of your application, and for any hobby projects, you can easily host them for free.
At the moment, the simplest (and free!) approach is to use Binder.  

According to the book, to publish our web app on Binder, we need to follow these steps: 
1. Add your notebook to a GitHub repository.  
*(We did this already in our previous topic when followed the steps provided in the forum by member vikbehal)*.   
We only need 3 files (+1 README.md that Github created when you created the new repo) to build this app:
- export.pkl, 
- *yournotebokkname*.ipynb 
- and requirements.txt
2. Paste the URL of that repo into Binder’s URL field. -As (1) in the image below-
3. Change the File drop-down to instead select URL.  
4. In the “URL to open” field, enter /voila/render/name.ipynb (replacing name with the name of your notebook). -As (2) in the image below-
5. Click the clipboard button at the bottom right to copy the URL - as (3) in the image below - and paste it somewhere safe.  
6. Click Launch.

![BinderSetUp](https://github.com/luisenoz/luisenoz.github.io/blob/master/images/binderlaunch.png)

The first time you do this, Binder will take around 5 minutes to build your site.  
Behind the scenes, it is finding a virtual machine that can run your app, allocating storage, and collecting the files needed for Jupyter, 
for your notebook, and for presenting your notebook as a web application.  
Finally, once it has started the app running, it will navigate your browser to your new web app.  
You can share the URL you copied to allow others to access your app as well.

> Since I couldn't make the notebbok run as a Voila web app, I haven't been able to test all this last part and deploy the app in Binder. But I promise you I'll do it some tiime in the future and I'll come back here to report my experience. *(I hope you have a much better luck!)*



