# Read and augmentation functions -----------------------------------------------------
## MDH add to read full image for infinite chip generator
fullImagesRead <- function(image_file,
                           mask_file) {
  img <- image_read(image_file)
  mask <- image_read(mask_file)
  return(list(img = img, mask = mask))
}

imagesRead <- function(image_file,
                       mask_file,
                       target_width = 128, 
                       target_height = 128) {
  img <- image_read(image_file)
  img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
  
  mask <- image_read(mask_file)
  mask <- image_scale(mask, paste0(target_width, "x", target_height, "!"))
  return(list(img = img, mask = mask))
}

randomBSH <- function(img,
                      u = 0,
                      brightness_shift_lim = c(90, 110), # percentage
                      saturation_shift_lim = c(95, 105), # of current value
                      hue_shift_lim = c(80, 120)) {
  
  if (rnorm(1) < u) return(img)
  
  brightness_shift <- runif(1, 
                            brightness_shift_lim[1], 
                            brightness_shift_lim[2])
  saturation_shift <- runif(1, 
                            saturation_shift_lim[1], 
                            saturation_shift_lim[2])
  hue_shift <- runif(1, 
                     hue_shift_lim[1], 
                     hue_shift_lim[2])
  
  img <- image_modulate(img, 
                        brightness = brightness_shift, 
                        saturation =  saturation_shift, 
                        hue = hue_shift)
  img
}

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
                                     batch_size) {
  # read image once.
  x_y_imgs_FULL <- fullImagesRead(image_file = image_path,
                                  mask_file = mask_path)
  img_x_dim <- image_info(x_y_imgs_FULL$img)$width
  img_y_dim <- image_info(x_y_imgs_FULL$img)$height
  
  # do the same for mask
  
  function() {
    
    # now loop over image and mask for 1:batch_size
    x_y_batch <- foreach(i = 1:batch_size) %dopar% {
      
      ### EXTRACT TILE HERE
      rnd_x_UL <- sample(0:img_x_dim-image_size,1)
      rnd_y_UL <- sample(0:img_y_dim-image_size,1)
      
      # could be done as apply function probably, doing this for now
      ### geometry string = "width x height + width offset + height offset" all from upper-left corner
      x_chip <- magick::image_crop(x_y_imgs_FULL$img, 
                                   paste0(image_size,"x",image_size,"+",rnd_x_UL,"+",rnd_y_UL))
      y_chip <- magick::image_crop(x_y_imgs_FULL$mask, 
                                   paste0(image_size,"x",image_size,"+",rnd_x_UL,"+",rnd_y_UL))
      
      # augmentation
      #x_y_imgs$img <- randomBSH(x_y_imgs$img)
      
      # return as arrays
      x_y_arr <- list(x = img2arr(x_chip),
                      y = mask2arr(y_chip))
    }
    
    x_y_batch <- purrr::transpose(x_y_batch)
    
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    
    result <- list(keras_array(x_batch), 
                   keras_array(y_batch))
    return(result)
  }
}


train_generator <- function(images_dir, 
                            samples_index,
                            masks_dir, 
                            batch_size,
                            class_name) {
  images_iter <- list.files(images_dir, 
                            pattern = ".tif", 
                            full.names = TRUE)[samples_index] # for current epoch
  images_all <- list.files(images_dir, 
                           pattern = ".tif",
                           full.names = TRUE)[samples_index]  # for next epoch
  masks_iter <- list.files(masks_dir, 
                           pattern = paste0(class_name,".tif"),
                           full.names = TRUE)[samples_index] # for current epoch
  masks_all <- list.files(masks_dir, 
                          pattern = paste0(class_name,".tif"),
                          full.names = TRUE)[samples_index] # for next epoch
  
  function() {
    
    # start new epoch
    if (length(images_iter) < batch_size) {
      images_iter <<- images_all
      masks_iter <<- masks_all
    }
    
    batch_ind <- sample(1:length(images_iter), batch_size)
    
    batch_images_list <- images_iter[batch_ind]
    images_iter <<- images_iter[-batch_ind]
    batch_masks_list <- masks_iter[batch_ind]
    masks_iter <<- masks_iter[-batch_ind]
    
    
    x_y_batch <- foreach(i = 1:batch_size) %dopar% {
      x_y_imgs <- imagesRead(image_file = batch_images_list[i],
                             mask_file = batch_masks_list[i])
      # augmentation
      x_y_imgs$img <- randomBSH(x_y_imgs$img)
      # return as arrays
      x_y_arr <- list(x = img2arr(x_y_imgs$img),
                      y = mask2arr(x_y_imgs$mask))
    }
    
    x_y_batch <- purrr::transpose(x_y_batch)
    
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    
    result <- list(keras_array(x_batch), 
                   keras_array(y_batch))
    return(result)
  }
}

val_generator <- function(images_dir, 
                          samples_index,
                          masks_dir, 
                          batch_size,
                          class_name) {
  images_iter <- list.files(images_dir, 
                            pattern = ".tif", 
                            full.names = TRUE)[samples_index] # for current epoch
  images_all <- list.files(images_dir, 
                           pattern = ".tif",
                           full.names = TRUE)[samples_index]  # for next epoch
  masks_iter <- list.files(masks_dir, 
                           pattern = paste0(class_name,".tif"),
                           full.names = TRUE)[samples_index] # for current epoch
  masks_all <- list.files(masks_dir, 
                          pattern = paste0(class_name,".tif"),
                          full.names = TRUE)[samples_index] # for next epoch
  
  function() {
    
    # start new epoch
    if (length(images_iter) < batch_size) {
      images_iter <<- images_all
      masks_iter <<- masks_all
    }
    
    batch_ind <- sample(1:length(images_iter), batch_size)
    
    batch_images_list <- images_iter[batch_ind]
    images_iter <<- images_iter[-batch_ind]
    batch_masks_list <- masks_iter[batch_ind]
    masks_iter <<- masks_iter[-batch_ind]
    
    
    x_y_batch <- foreach(i = 1:batch_size) %dopar% {
      x_y_imgs <- imagesRead(image_file = batch_images_list[i],
                             mask_file = batch_masks_list[i])
      # without augmentation
      
      # return as arrays
      x_y_arr <- list(x = img2arr(x_y_imgs$img),
                      y = mask2arr(x_y_imgs$mask))
    }
    
    x_y_batch <- purrr::transpose(x_y_batch)
    
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    
    result <- list(keras_array(x_batch), 
                   keras_array(y_batch))
    return(result)
  }
}
