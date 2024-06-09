
# Image-Caption-Generator

In this project , we will build an image caption generator to load a random image and give some captions describing the image. We will use Convolutional Neural Network (CNN) for image feature extraction and Long Short-Term Memory Network (LSTM) for Natural Language Processing (NLP).


## Dataset Information

The objective of the project is to predict the captions for the input image. The dataset consists of 8k images and 5 captions for each image. The features are extracted from both the image and the text captions for input. 


The features will be concatenated to predict the next word of the caption. CNN is used for image and LSTM is used for text. BLEU Score is used as a metric to evaluate the performance of the trained model.

Download the Flickr dataset here
## Extract Image Features

Used VGG16() model fot image features extraction.Fully connected layer of the VGG16 model is not needed, just the previous layers to extract feature results.

Dictionary 'features' is created and will be loaded with the extracted features of image data

load_img(img_path, target_size=(224, 224)) - custom dimension to resize the image when loaded to the array

image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) - reshaping the image data to preprocess in a RGB type image.

model.predict(image, verbose=0) - extraction of features from the image

img_name.split('.')[0] - split of the image name from the extension to load only the image name.

## Preprocess Text Data

Defined to clean and convert the text for quicker process and better results.

**Example:** 

['A child in a pink dress is climbing up a set of stairs in an entry way .',
 'A girl going into a wooden building .',
 'A little girl climbing into a wooden playhouse .',
 'A little girl climbing the stairs to her playhouse .',
 'A little girl in a pink dress going into a wooden cabin .']

After cleaning:

['startseq child in pink dress is climbing up set of stairs in an entry way endseq',
 'startseq girl going into wooden building endseq',
 'startseq little girl climbing into wooden playhouse endseq',
 'startseq little girl climbing the stairs to her playhouse endseq',
 'startseq little girl in pink dress going into wooden cabin endseq']


Words with one letter was deleted

All special characters were deleted

'startseq' and 'endseq' tags were added to indicate the start and end of a caption for easier processing
## Model Information

![App Screenshot](https://github.com/Jerrinthomas007/images/blob/0dae38765bc1bda39133f04b4e2fac7167db4298/image%20caption%20model.png)

**shape**=(4096,)- output length of the features from the VGG model

**Dense** - single dimension linear layer array

**Dropout()** - used to add regularization to the data, avoiding over fitting & dropping out a fraction of the data from the layers

**model.compile()** - compilation of the model

**loss=’sparse_categorical_crossentropy’** - loss function for category outputs

**optimizer=’adam’** - automatically adjust the learning rate for the model over the no. of epochs

Model plot shows the concatenation of the inputs and outputs into a single layer

Feature extraction of image was already done using VGG, no CNN model was needed in this step.
## Learnings

Training the model by increasing the no. of epochs can give better and more accurate results.

Processing large amount of data can take a lot of time and system resource.

The no. of layers of the model can be increased if you want to process large dataset like flickr32k.

Instead of VGG16, we can also use CNN model to extract the image features.
