# From a trained model to an Online Application!

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
5. Make preditions:
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
The books offer a notebook called "bear classifier" that we can copy and update to our set of data, with minimun chnages.

Next, we need to install Voilà by copying these lines into a notebook cell and executing it:
```python
!pip install voila
!jupyter serverextension enable voila --sys-prefix
```
The first line installs the voila library and application, and the second connects it to your existing Jupyter notebook.

Voilà runs Jupyter notebooks just like the Jupyter notebook server we you are using, but it removes all of the cell inputs, 
and shows only output (including ipywidgets), along with any Markdown cells. <ins>So what’s left is a web application!</ins>. 

The book sys that to view the notebook as a Voilà web application, we need to replace the word “notebooks” in our browser’s URL with “voila/render”.  
However, I get an *404 : Not Found - You are requesting a page that does not exist!* message, instead of a Voila web app.  
I'll look into it in th fastai forum.

