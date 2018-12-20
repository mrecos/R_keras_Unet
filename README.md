# R_keras_Unet
Unet implementation in R flavor of Keras


## Semantic segmentation on aerial imagery


### Buildings
![Alt text](images/example_prob.png?raw=true "Title") ![Alt text](images/example_mask.png?raw=true "Title")


### Roads
![Alt text](images/Roads_prob.png?raw=true "Title") ![Alt text](images/roads_mask.png?raw=true "Title")

## infinite training chip generator

## on-the-fly image and mask augmentation
![Alt text](images/augment1.png?raw=true "Title") ![Alt text](images/augment2.png?raw=true "Title") ![Alt text](images/augment3.png?raw=true "Title")


## TODO:

* Need image loader to remove or sample from chips that have no target in them. They could be out of the study area or just lacking a thing.  For rare classes, this could be many/most random tiles. might want to think of better way.
* somehow split out training image vs val image
  * add stride/padding/offset and train/val split values to config
  * add function to make all possible overlapping tile UL coords
  * split all-possible into train/val based on config value
  * only pull infinite chip from train list of possible UL corner coords
  * has to check to see if all-possble coords exist, but DOES NOT KNOW if you changed dim, need to make it aware
    * ran into this bug before b/c no notice of using old data. add message when ussing old data
  * add option to config to reser the all-possble train/val splits
  * has to be able to also function as it does now where split% == 0 so train/val randomly from same of seperate study area images
  * Alternative to random select of train/val set (best in Kfolds CV) is to have full train/val images or either seperate geographies or strips/sections/blocks of single geography where train is NA where Val has data.
  
* add functionality to record dice/jaccard by batch and chip coords to reconstruct error map
  * somehow write out coords form batch and then join to error metric of that batch.
  * did this with a custom callback, but only records training dice coef because validation is only at end of epoch
  * create second validation iterator to get coords of validation after model fit
    * post-model-fit validation loop for mapping moved into own script. 
    * Then work on train/val all-possible coords lists to use instead...
    * Need to figure out mapping pixel coords to geographic coords
  
* image normalization to dataset mean/variance (in config)

* Test additional architectures

* Test Time Augmenting, flipping, mirror, etc... (large area prediction)

* Make it easy to test on different classes (still all binary class prediction)



## Info/Sources


* Rstudio Keras example repo of Unet
  * https://github.com/rstudio/keras/blob/master/vignettes/examples/unet.R
  
* Augment images and masks
  * https://github.com/keras-team/keras/issues/3059
  * https://keras.io/preprocessing/image/
  * https://github.com/rstudio/keras/blob/707427fa1f8e192e662b209e63fb962e99ba8fbf/vignettes/examples/cifar10_cnn.R
  * https://keras.rstudio.com/reference/image_data_generator.html

* fit_generator pulling too many images from data generator because workers >= 1
  * https://stackoverflow.com/questions/49267020/keras-flow-from-directory-saving-too-many-images-during-fit-generator-call
  
* Custom callback to record metrics as end of each batch
  * https://keras.rstudio.com/articles/training_callbacks.html

* Memory issues with magick package. Cannot load full SARA raster.
  * https://github.com/ImageMagick/ImageMagick/issues/396
  

  