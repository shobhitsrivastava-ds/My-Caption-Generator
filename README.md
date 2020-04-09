# My-Caption-Generator
## Brief Overview :

* You saw an image and your brain can easily tell what the image is about, but can a computer tell what the image is representing? Computer vision researchers worked on this a lot and they considered it impossible until now! With the advancement in Deep learning techniques, availability of huge datasets and computer power, we can build models that can generate captions for an image.

* This is what we are going to implement in this Python based project where we will use deep learning techniques of Convolutional Neural Networks and a type of Recurrent Neural Network (LSTM) together.
<hr>
<div style="text-align:center"><img src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/11/working-of-Deep-CNN-Python-project.png" /></div>
<hr>

### What is Image Caption Generator?
<p>Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.</p>

### Image Caption Generator with CNN – About the Python based Project
<p>The objective of our project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.<p>

* In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

* The Dataset of Python based Project
For the image caption generator, we will be using the Flickr_8K dataset. There are also other big datasets like Flickr_30K and MSCOCO dataset but it can take weeks just to train the network so we will be using a small Flickr8k dataset. The advantage of a huge dataset is that we can build better models.

###### Thanks to Jason Brownlee for providing a direct link to download the dataset (Size: 1GB).

*Flicker8k_Dataset 
*Flickr_8k_text 
#### The Flickr_8k_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions separated by newline(“\n”).

### Pre-requisites
* This project requires good knowledge of Deep learning, Python, working on Jupyter notebooks, Keras library, Numpy, and Natural language processing.

#### Make sure you have installed all the following necessary libraries:

* pip install tensorflow
* keras
* pillow
* numpy
* tqdm
* jupyterlab
* Image Caption Generator – Python based Project
