# CassavaLeafDisease

Kaggle Profile: https://www.kaggle.com/brendanartley

Multi-label computer vision classification model of diseased cassava plants. The following code files are how I created my model to submit to the Cassava Leaf Disease Competition hosted by Kaggle. I encourage you to check out the files on Kaggle for increased readability and different model versions. I have included an example of a typical image that would be classified below.

![Cassava image](https://github.com/brendanartley/cassava_leaf_disease/blob/main/cld_example.png)

## tpu-training-notebook

This is the main notebook where I created and trained models using Google's TPU cores. I attempted many different model types, data augmentation techniques, custom loss functions, and more. When improvements were made on the model accuracy I included these in the notes section at the top of the notebook. 

The final version of the notebook is a five-fold model ensemble of efficient-net-b3's using center-cropped 512 x 512 images stored as tfrecords. Training this model on TPU's allowed the entire notebook to run in 2 - 2.5hrs. I used standard categorical cross-entropy with label smoothing, and added average pooling and dropout layers on the models to reduce the likelihood of overfitting. Another important thing to note was that I created a custom callback function that updated the learning rate by each step rather than by each epoch.

To see previous versions and experimentations feel free to check them out on Kaggle. 

## creating-tfrecords

This notebook was the reason why I was able to experiment with so many different parameters, callbacks, loss functions, and training cycles. Storing the images as tfrecords and training on TPU's was more than 8x faster than training the exact same model on GPU's. I also got a large boost in model performance when I added additional images of cassava plants from a previous competition. I made sure to do a stratified split to get an equal balance of each class in every tf.data file. 

## gpu_training_baseline

I initially started this project by training some basic single models using GPU's. Using TensorFlow's built-in image data generator I was able to try out some simple data augmentation techniques and get a sense of what would work for this dataset. I also used this notebook to visualize some examples of the images that were in each class. I used the model created in this notebook as the baseline for this project.

## cld-prediction

This notebook is pretty simple. It just shows how I was using my model to make predictions on unseen data in the kaggle notebook environment.

## cld-noisy-labels

This notebook is an attempt at relabelling noisy images from the dataset. I did quite a bit of reading and looking into how to deal with noisy images, yet couldn't really get it to work. If I had more time on this project I would train a model on a very clean dataset and then relabel or remove images if predictions were extremely in the wrong direction. These ideas stem from this presentation on Youtube. https://www.youtube.com/watch?v=8mpBHbjG4E4
