# 1. Packages / Options ---------------------------------------------------

if (!require(keras)) install.packages('keras'); require(keras)
if (!require(grid)) install.packages('grid'); require(grid)
if (!require(gridExtra)) install.packages('gridExtra'); require(gridExtra)
if (!require(magick)) install.packages('magick'); require(magick)
if (!require(viridis)) install.packages('viridis'); require(viridis)

# Pre-Trained Model(CAD_small) 

model <- load_model_hdf5('C:\\Users\\Mano\\Desktop\\R_keras\\Chap5\\CAD_small.h5',
                         compile = F)

model

# 2. Activations Visualization --------------------------------------------

img_path <- 'C:\\Users\\Mano\\Desktop\\dogs-vs-cats\\small\\test\\cats\\cat.1700.jpg'

img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255

plot(as.raster(img_tensor[1,,,]))


layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)


plot_channel <- function(channel){
  
  rotate <- function(x){t(apply(x, 2, rev))}
  image(rotate(channel), axes = F, asp = 1, col = terrain.colors(12))

  }

plot_channel(first_layer_activation[1,,,2])


image_size <- 58
images_per_row <- 16

for(i in 1:8){
  
  activation <- activations[[i]]
  name <- model$layers[[i]]$name
  
  n_features <- dim(activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0('cat_activations_', i, '_', name, '.png'),
      width = image_size * images_per_row,
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(.02, 4))
  
  for(col in 0:(n_cols-1)){
    for(row in 0:(images_per_row-1)){
      
      channel_image <- activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
      
    }
  }
  
  par(op)
  dev.off()
}



# 3. Filter Visualization -------------------------------------------------

model <- application_vgg16(
  
  weights = 'imagenet',
  include_top = F
  
  )

deprocess_image <- function(x){
  
  dms <- dim(x)
  
  x <- x - mean(x)
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1
  
  x <- x + 0.5
  x <- pmax(0, pmin(x, 1))
  
  array(x, dim = dms)
}

generate_pattern <- function(layer_name, filter_index, size = 150){
  
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index])
  
  grads <- k_gradients(loss, model$input)[[1]]
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  iterate <- k_function(list(model$input), list(loss, grads))
  
  input_img_data <- array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  step <- 1
  for(i in 1:40){
    
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step)
    
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img)
  
  }

grid.raster(generate_pattern('block3_conv1', 1))

# 4. Class Activation Map(CAM Visualization) ------------------------------

model <- application_vgg16(weights = 'imagenet')

img_path <- 'C:\\Users\\Mano\\Desktop\\dogs-vs-cats\\elephant.png'

img <- image_load(img_path, target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  imagenet_preprocess_input()

preds <- model %>% predict(img)
imagenet_decode_predictions(preds)
which.max(preds[1,]) # 387 = African Elephant

african_elephant_output <- model$output[, 387]
last_conv_layer <- model %>% get_layer('block5_conv3')
grads <- k_gradients(african_elephant_output, last_conv_layer$output)[[1]]

pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
iterate <- k_function(list(model$input), list(pooled_grads, last_conv_layer$output[1,,,]))

c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
for(i in 1:512){
  
  conv_layer_output_value[,,i] <- conv_layer_output_value[,,i] * pooled_grads_value[[i]]

  }

heatmap <- apply(conv_layer_output_value, c(1,2), mean)
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)

write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = 'white', col = terrain.colors(12)){
  
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0, 0, 0, 0))
  on.exit({par(op); dev.off()}, add = T)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = F, asp = 1, col = col)
  
}

write_heatmap(heatmap, 'elephant_heatmap.jpg')

image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf('%dx%d!', info$width, info$height)

pal <- col2rgb(viridis(20), alpha = T)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, 'elephant_overlay.png', bg = NA, col = pal_col)

image_read('elephant_overlay.png') %>% 
  image_resize(geometry = geometry, filter = 'quadratic') %>% 
  image_composite(image, operator = 'blend', compose_args = '20') %>% 
  plot()

