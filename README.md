# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for a flower image classifier built with PyTorch, then convert it into a command line application.


## First part: 
### Develop code for an image classifier

A Python application has been developed, that can train an image classifier on a dataset, then predict new images using the trained model.
See Image Classifier Project.ipynb Jupyter notebook.
 
## Second part:
### Create a command line application
The application developed in the first part of the course is modified to create a command line application. 
Use train.py and predict.py to train model and make predictions. 
Modelutils.py contains all functions related to the model. Utils.py contains extra helper functions. The commmand line arguments are defined in inargs.py.

  
  
##### train.py: 
> trains the neural network with a set of images

command line arguments:
```
data_dir, directory of the train images, no default value
--save_dir, default = ./, directory for saving the checkpoint
--arch, default = vgg16, architecture of the model
--learning_rate, default = 0.001
--hidden_units, default = 4096, the number of hidden units for the classifier
--epochs, default = 10
--gpu, default = False, use cuda or not?
```    
    
##### predict.py: 
> predicts the type of a flower based on an input image

command line arguments:
```
input, input image for flower type prediction, no default value
checkpoint, path to checkpoint.pth file, no default value
--top_k, default  = 5, top K predicted classes
--category_names, default = cat_to_name.json, Json file mapping file names to category names
--gpu, default=False, use cuda or not?
--bar_chart, default=False, view bar plot with probabilities, or only print results?'
```
