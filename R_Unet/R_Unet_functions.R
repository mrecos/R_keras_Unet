# Read and augmentation functions -----------------------------------------------------
## MDH add to read full image for infinite chip generator
fullImagesRead <- function(image_file,
                           mask_file) {
  img <- image_read(image_file)
  mask <- image_read(mask_file)
  return(list(img = img, mask = mask))
}

## MDH input x (image) & y (mask) tensor batch, augment based on augment_args, return arugmented x & y batches
augment_tensor <- function(x_batch, y_batch, augment_args){
  # we create two instances with the same arguments
  seed = 1L ### `L` is critical! and both having the same seed is critical!
  # Create identical augmenting generators for image and mask
  image_datagen = do.call(image_data_generator, augment_args)
  mask_datagen  = do.call(image_data_generator, augment_args)
  # create iterator of our batch from generators
  image_generator <- flow_images_from_data(x_batch, generator = image_datagen, seed=seed)
  mask_generator  <- flow_images_from_data(y_batch, generator = mask_datagen,  seed=seed)
  # generate new batches from iterator
  x_batch <- generator_next(image_generator)
  y_batch <- generator_next(mask_generator)
  return(list(x_batch = x_batch, y_batch = y_batch))
}

# imagesRead <- function(image_file,
#                        mask_file,
#                        target_width = 128, 
#                        target_height = 128) {
#   img <- image_read(image_file)
#   img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
#   
#   mask <- image_read(mask_file)
#   mask <- image_scale(mask, paste0(target_width, "x", target_height, "!"))
#   return(list(img = img, mask = mask))
# }

img2arr <- function(image, 
                    target_width = 128,
                    target_height = 128) {
  result <- aperm(as.numeric(image[[1]])[, , 1:3], c(2, 1, 3)) # transpose
  array_reshape(result, c(1, target_width, target_height, 3))
}

mask2arr <- function(mask,
                     target_width = 128,
                     target_height = 128) {
  result <- t(as.numeric(mask[[1]])[, , 1]) # transpose
  array_reshape(result, c(1, target_width, target_height, 1))
}

## MDH - function for creating infinite chip generator for Keras
train_infinite_generator <- function(image_path, 
                                     mask_path, 
                                     image_size,
                                     batch_size,
                                     amt_train,
                                     epochs = 1,
                                     use_augmentation = FALSE,
                                     augment_args = NULL,
                                     mode = c("train","validate","predict")) {
  ## error catch
  if(isTRUE(use_augmentation)){
    try(if(is.null(augment_args)) stop("Error: Must supply a list of augmentation arguments."))
  }
  
  ## create log to record selected chip UL-corner coords (coul dbe moved to a make coord log function IF TRUE)
  coord_file_name <- paste0("./logs_r/","coordinates_",mode,"_",format(Sys.time(), "%M-%H_%d_%m_%Y"),".csv")
  file.create(coord_file_name)
  write(paste("x_coords","y_coord","sample","step","epoch","mode",sep=","), file = coord_file_name, append = TRUE)
  
  # FULL read image and mask once
  x_y_imgs_FULL <- fullImagesRead(image_file = image_path,
                                  mask_file = mask_path)
  img_x_dim <- image_info(x_y_imgs_FULL$img)$width
  img_y_dim <- image_info(x_y_imgs_FULL$img)$height
  
  step_counter = 0 # prepare counter
  steps_per_epoch = (amt_train/batch_size) # calculate steps per batch
  function() {
    ## some counters to keep track of coordinate pairs.
    step_counter <<- step_counter + 1 # use global assign to increment each call to this function
    step  <- step_counter %% steps_per_epoch # keep number of steps 
    step  <- ifelse(step == 0, steps_per_epoch, step) # fix case when counter == steps
    epoch <- ceiling(step_counter / steps_per_epoch) # increment of epochs counter
    
    # now loop over image and mask for 1:batch_size
    x_y_batch <- vector(mode = "list", length = batch_size)
    #x_y_batch <- foreach(i = 1:batch_size) %dopar% {  # DOPAR, may be issue in future
    for(i in seq_len(batch_size)){
      
      ### Random sample for chip upper-left corner image coordinates
      rnd_x_UL <- sample(0:img_x_dim-image_size,1)
      rnd_y_UL <- sample(0:img_y_dim-image_size,1)
      
      # write selected coordinates to log
      #cat("\n",paste(rnd_x_UL,rnd_y_UL,i,step,epoch,mode,sep=","),"\n") # for testing
      write(paste(rnd_x_UL,rnd_y_UL,i,step,epoch,mode,sep=","), file = coord_file_name, append = TRUE)
      
      # Extract chip from FULL image and mask (using same coordinates for both)
      ### geometry string = "width x height + width offset + height offset" all from upper-left corner
      x_chip <- magick::image_crop(x_y_imgs_FULL$img, paste0(image_size,"x",image_size,"+",rnd_x_UL,"+",rnd_y_UL))
      y_chip <- magick::image_crop(x_y_imgs_FULL$mask, paste0(image_size,"x",image_size,"+",rnd_x_UL,"+",rnd_y_UL))
      
      # Could do some form of pure image augmentation here (as opposed to with keras augmentor)
      
      # create as arrays
      x_y_arr <- list(x = img2arr(x_chip, target_width=image_size, target_height=image_size),
                      y = mask2arr(y_chip, target_width=image_size, target_height=image_size))
      # add to batch results (modified below out of loop)
      x_y_batch[[i]] <- x_y_arr
    }
    
    # reshape image matrices into list and the contatenate into x and y
    x_y_batch <- purrr::transpose(x_y_batch)
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    
    if(isTRUE(use_augmentation)){
      augmented_batch <- augment_tensor(x_batch, y_batch, augment_args)
      x_batch <- augmented_batch$x_batch
      y_batch <- augmented_batch$y_batch
    }
    
    result <- list(keras_array(x_batch), 
                   keras_array(y_batch))
    return(result)
  }
}



# train_generator <- function(images_dir, 
#                             samples_index,
#                             masks_dir, 
#                             batch_size,
#                             class_name) {
#   images_iter <- list.files(images_dir, 
#                             pattern = ".tif", 
#                             full.names = TRUE)[samples_index] # for current epoch
#   images_all <- list.files(images_dir, 
#                            pattern = ".tif",
#                            full.names = TRUE)[samples_index]  # for next epoch
#   masks_iter <- list.files(masks_dir, 
#                            pattern = paste0(class_name,".tif"),
#                            full.names = TRUE)[samples_index] # for current epoch
#   masks_all <- list.files(masks_dir, 
#                           pattern = paste0(class_name,".tif"),
#                           full.names = TRUE)[samples_index] # for next epoch
#   
#   function() {
#     
#     # start new epoch
#     if (length(images_iter) < batch_size) {
#       images_iter <<- images_all
#       masks_iter <<- masks_all
#     }
#     
#     batch_ind <- sample(1:length(images_iter), batch_size)
#     
#     batch_images_list <- images_iter[batch_ind]
#     images_iter <<- images_iter[-batch_ind]
#     batch_masks_list <- masks_iter[batch_ind]
#     masks_iter <<- masks_iter[-batch_ind]
#     
#     
#     x_y_batch <- foreach(i = 1:batch_size) %dopar% {
#       x_y_imgs <- imagesRead(image_file = batch_images_list[i],
#                              mask_file = batch_masks_list[i])
#       # augmentation
#       x_y_imgs$img <- randomBSH(x_y_imgs$img)
#       # return as arrays
#       x_y_arr <- list(x = img2arr(x_y_imgs$img),
#                       y = mask2arr(x_y_imgs$mask))
#     }
#     
#     x_y_batch <- purrr::transpose(x_y_batch)
#     
#     x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
#     
#     y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
#     
#     result <- list(keras_array(x_batch), 
#                    keras_array(y_batch))
#     return(result)
#   }
# }
# 
# val_generator <- function(images_dir, 
#                           samples_index,
#                           masks_dir, 
#                           batch_size,
#                           class_name) {
#   images_iter <- list.files(images_dir, 
#                             pattern = ".tif", 
#                             full.names = TRUE)[samples_index] # for current epoch
#   images_all <- list.files(images_dir, 
#                            pattern = ".tif",
#                            full.names = TRUE)[samples_index]  # for next epoch
#   masks_iter <- list.files(masks_dir, 
#                            pattern = paste0(class_name,".tif"),
#                            full.names = TRUE)[samples_index] # for current epoch
#   masks_all <- list.files(masks_dir, 
#                           pattern = paste0(class_name,".tif"),
#                           full.names = TRUE)[samples_index] # for next epoch
#   
#   function() {
#     
#     # start new epoch
#     if (length(images_iter) < batch_size) {
#       images_iter <<- images_all
#       masks_iter <<- masks_all
#     }
#     
#     batch_ind <- sample(1:length(images_iter), batch_size)
#     
#     batch_images_list <- images_iter[batch_ind]
#     images_iter <<- images_iter[-batch_ind]
#     batch_masks_list <- masks_iter[batch_ind]
#     masks_iter <<- masks_iter[-batch_ind]
#     
#     
#     x_y_batch <- foreach(i = 1:batch_size) %dopar% {
#       x_y_imgs <- imagesRead(image_file = batch_images_list[i],
#                              mask_file = batch_masks_list[i])
#       # without augmentation
#       
#       # return as arrays
#       x_y_arr <- list(x = img2arr(x_y_imgs$img),
#                       y = mask2arr(x_y_imgs$mask))
#     }
#     
#     x_y_batch <- purrr::transpose(x_y_batch)
#     
#     x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
#     
#     y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
#     
#     result <- list(keras_array(x_batch), 
#                    keras_array(y_batch))
#     return(result)
#   }
# }

#  Plotting functions ---------------------------------------------

plot_pred_tensor_overlay <- function(prediction, actual, indx = 1, cnfg = config, 
                                     alpha = 0.5, mask = FALSE, mask_threshold = 0.5,
                                     viridis_option = "D", use_legend = TRUE){
  library("ggplot2")
  library("raster")
  library("grid")  
  
  pred   <- prediction[indx,,,]
  actual <- actual[indx,,,]
  
  pred_df <- raster::raster(pred, xmn=0, xmx=cnfg$IMAGE_SIZE, 
                            ymn=0, ymx=cnfg$IMAGE_SIZE)
  pred_df <- as(pred_df, "SpatialPixelsDataFrame")
  pred_df <- as.data.frame(pred_df)
  
  if(isTRUE(mask)){
    pred_df[which(pred_df$layer >= mask_threshold),"layer"] <- 1
    pred_df[which(pred_df$layer < mask_threshold),"layer"]  <- NA
  }
  if(isTRUE(use_legend)){
    legend_pos = "bottom"
  } else {
    legend_pos = "none"
  }
  
  p <- ggplot() +  
    annotation_custom(grid::rasterGrob(actual),
                      xmin=0, xmax=cnfg$IMAGE_SIZE,
                      ymin=0, ymax=cnfg$IMAGE_SIZE) +
    geom_tile(data=pred_df, aes(x=x, y=y, fill=layer), alpha=alpha) + 
    scale_fill_viridis_c(name="prob", na.value=NA, limits = c(0,1), option = viridis_option) +
    coord_equal() +
    theme_void() +
    theme(legend.position=legend_pos) +
    theme(legend.key.width=unit(2, "cm"))
  return(p)
}
# Custom error metric logger -----------------------------------------
# MDH - based on example here: # https://keras.rstudio.com/articles/training_callbacks.html
# Records the training dice coef for each batch/step in training
# cannot record validation dice coef b/c validation is only at end of epoch not each batch.
dice_coef_by_batch <- R6::R6Class("DiceHistory",
                                  inherit = KerasCallback,
                                  
                                  public = list(
                                    
                                    dice_coef = NULL,
                                    
                                    on_batch_end = function(batch, logs = list()) {
                                      self$dice_coef <- c(self$dice_coef, logs[["dice_coef"]])
                                    }
                                  ))
