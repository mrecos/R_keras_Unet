# R_keras_Unet
Unet implementation in R flavor of Keras


## Starting point for a translating the unet CNN to R 
![Alt text](images/example_mask.png?raw=true "Title")
![Alt text](images/example_prob.png?raw=true "Title")


## TODO:


* Image augmentation

* somehow split out training image vs val image

* Test additional archtecture

* Test Time Augmenting, flipping, mirror, etc... (large area prediction)

* Make it easy to test on different classes (still all binary class prediction)



## Info/Sources


* Rstudio Keras example repo of Unet
  * https://github.com/rstudio/keras/blob/master/vignettes/examples/unet.R
  
* Augment images and masks
  * https://github.com/keras-team/keras/issues/3059
  * https://keras.io/preprocessing/image/
  * https://github.com/rstudio/keras/blob/707427fa1f8e192e662b209e63fb962e99ba8fbf/vignettes/examples/cifar10_cnn.R
  