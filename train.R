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


#### TESTING AREA train_generator step through
### testing infinite generator
inf_test <- train_infinite_generator(image_path = config$TRAIN_IMG,
                                     mask_path  = config$TRAIN_MSK,
                                     image_size = config$IMAGE_SIZE,
                                     batch_size = config$BATCH_SIZE)
inf_batch <- inf_test()
image_test <- as.matrix(inf_batch[[1]])
plot(as.raster(image_test[1,,,]))
# rasterImage(as.raster(image_test[1,,,]),0,0,128,128)
mask_test <- as.matrix(inf_batch[[2]])
plot(as.raster(mask_test[1,,,]))
# rasterImage(as.raster(mask_test[1,,,]),0,0,128,128, alpha = 0.5)
###### END TESTING ###


# Create Interators ---------------------------------------------
train_infinite_iterator <- py_iterator(train_infinite_generator(image_path = config$TRAIN_IMG,
                                                                mask_path  = config$TRAIN_MSK,
                                                                image_size = config$IMAGE_SIZE,
                                                                batch_size = config$BATCH_SIZE))
predict_generator <- train_infinite_generator(image_path = config$TRAIN_IMG,
                                              mask_path  = config$TRAIN_MSK,
                                              image_size = config$IMAGE_SIZE,
                                              batch_size = 10)
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

# fit with infinite generator
history <- model %>% fit_generator(
  train_infinite_iterator,
  steps_per_epoch = as.integer(config$AMT_TRAIN / config$BATCH_SIZE), 
  epochs = config$EPOCHS, 
  validation_data = train_infinite_iterator,
  validation_steps = as.integer(config$AMT_TRAIN / config$BATCH_SIZE),
  verbose = 1,
  callbacks = callbacks_list
)

model %>% evaluate_generator(train_infinite_iterator, steps = 5)

predict_batch <- as.matrix(predict_generator()[[1]])
preds <- model %>% predict(predict_batch, steps = 1)
plot_pred_tensor_overlay(preds, predict_batch, 9, alpha = 0.45, mask=TRUE)

 

     