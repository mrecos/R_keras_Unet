##########################
# Title:  Fit Unet model on existing image tiles (chips)
# Inputs: config file, unet model, functions, test & train dirs for existing image tiles
# Ouputs: trained Unet model, logs, some visualization
# Status: Working / Put on hold / not developing
# Notes:  This is the first pass at training Unet with existing chips. But put on hold to work 
#         on the version with inifinite tile generator. All here should work if 
#         config file points to directories with chips.
##########################

library("config")

#library(keras)
library(magick)
library(abind)
#library(reticulate)
library(parallel)
library(doParallel)
library(foreach)

config <- config::get(config = "testing", file = "./R_Unet/R_Unet_config.yml")
source(file = "./R_Unet/R_Unet_functions.R")
source(file = "./R_Unet/R_Unet.R")

model <- get_unet_128(input_shape = c(config$IMAGE_SIZE, config$IMAGE_SIZE, config$N_CHANNELS),
                      num_classes = config$NUM_CLASSES)

#train_samples <- 5088
# not the right approach, but a band-aid while combining code
train_index <- sample(1:config$AMT_TRAIN, round(config$AMT_TRAIN * 0.8)) # 80%
val_index <- c(1:config$AMT_TRAIN)[-train_index]


#### TESTING AREA train_generator step through
images_iter <- list.files(config$TRAIN_IMG, 
                          pattern = ".tif", 
                          full.names = TRUE)[train_index] # for current epoch
images_all <- list.files(config$TRAIN_IMG, 
                         pattern = ".tif",
                         full.names = TRUE)[train_index]  # for next epoch
masks_iter <- list.files(config$TRAIN_MSK, 
                         pattern = paste0(config$CLASS_NAME,".tif"),
                         full.names = TRUE)[train_index] # for current epoch
masks_all <- list.files(config$TRAIN_MSK, 
                        pattern = paste0(config$CLASS_NAME,".tif"),
                        full.names = TRUE)[train_index] # for next epoch


# this is the generator process, works, but not respecting target class and other things
train_test <- train_generator(images_dir = config$TRAIN_IMG,
                masks_dir  = config$TRAIN_MSK,
                samples_index = train_index,
                batch_size = config$BATCH_SIZE,
                class_name = config$CLASS_NAME)
batch_test <- train_test()
image_test <- as.matrix(batch_test[[1]])
plot(as.raster(image_test[1,,,]))
# rasterImage(as.raster(image_test[1,,,]),0,0,128,128)
mask_test <- as.matrix(batch_test[[2]])
plot(as.raster(mask_test[1,,,]))
# rasterImage(as.raster(mask_test[1,,,]),0,0,128,128, alpha = 0.5)


train_iterator <- py_iterator(train_generator(images_dir = config$TRAIN_IMG,
                                              masks_dir  = config$TRAIN_MSK,
                                              samples_index = train_index,
                                              batch_size = config$BATCH_SIZE,
                                              class_name = config$CLASS_NAME))

val_iterator <- py_iterator(val_generator(images_dir = config$VAL_IMG,
                                          masks_dir = config$VAL_MSK,
                                          samples_index = val_index,
                                          batch_size = config$BATCH_SIZE,
                                          class_name = config$CLASS_NAME))


# Training -----------------------------------------------------

tensorboard("logs_r")

callbacks_list <- list(
  callback_tensorboard("logs_r"),
  callback_early_stopping(monitor = "val_python_function",
                          min_delta = 1e-4,
                          patience = 8,
                          verbose = 1,
                          mode = "max"),
  callback_reduce_lr_on_plateau(monitor = "val_python_function",
                                factor = 0.1,
                                patience = 4,
                                verbose = 1,
                                #epsilon = 1e-4,
                                mode = "max"),
  callback_model_checkpoint(filepath = "weights_r/unet128_{epoch:02d}.h5",
                            monitor = "val_python_function",
                            save_best_only = TRUE,
                            save_weights_only = TRUE, 
                            mode = "max" )
)

# fit with chip iterator
history <- model %>% fit_generator(
  train_iterator,
  steps_per_epoch = as.integer(length(train_index) / config$BATCH_SIZE),
  epochs = config$EPOCHS,
  validation_data = val_iterator,
  validation_steps = as.integer(length(val_index) / config$BATCH_SIZE),
  verbose = 1,
  callbacks = callbacks_list
)


model %>% evaluate_generator(val_iterator, steps = 5)

preds <- model %>% predict(batch_test[[1]], steps = 1)
plot(as.raster(preds[1,,,]))
