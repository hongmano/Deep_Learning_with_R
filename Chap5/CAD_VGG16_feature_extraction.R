# 1. Packages / Options ---------------------------------------------------

if (!require(keras)) install.packages('keras'); require(keras)

# Pre-Trained Model(VGG16) 

conv_base <- application_vgg16(
  weights = 'imagenet',
  include_top = F,
  input_shape = c(150, 150, 3)
)

# 2. Utils ----------------------------------------------------------------

base_dir <- 'C:/Users/Mano/Desktop/dogs-vs-cats/small'
train_dir <- file.path(base_dir, 'train')
validation_dir <- file.path(base_dir, 'validation')
test_dir <- file.path(base_dir, 'test')

# 3. Feature Extraction without Augmentation ------------------------------

datagen <- image_data_generator(rescale = 1/255)

batch_size <- 20

extract_features <- function(directory, sample_count){
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = 'binary'
  )
  
  i <- 0 
  while(T){
    
    batch <- generator_next(generator)
    
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      break
    
  }
  
  list(features = features,
       labels = labels)
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

# Flatten

reshape_features <- function(features){
  
  array_reshape(features, 
                dim = c(nrow(features), 4 * 4 * 512))
  
  }

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# Model

model1 <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = .5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model1 %>% compile(
  
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = loss_binary_crossentropy,
  metrics = metric_binary_accuracy

  )

# Fit

hitory1 <- model1 %>% fit(
  
  train$features,
  train$labels,
  epochs = 30, 
  batch_size = 20,
  validation_data = list(validation$features,
                         validation$labels)

  )

# 4. Feature Extraction with Augmentation ---------------------------------

train_datagen <- image_data_generator(
  
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = .2,
  height_shift_range = .2,
  shear_range = .2,
  zoom_range = .2,
  horizontal_flip = T,
  fill_mode = 'nearest'
  
  )

train_generator <- flow_images_from_directory(
  
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
  
  )

validation_generator <- flow_images_from_directory(
  
  validation_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
  
  )


# Model 

model2 <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model2 %>% compile(
  
  loss = loss_binary_crossentropy,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = metric_binary_accuracy
  
  )

model2

# Freezing

cat('Trainable weights before freezing : ', length(model2$trainable_weights))
freeze_weights(conv_base)
cat('Trainable weights after freezing : ', length(model2$trainable_weights))

# Fit

history2 <- model2 %>% fit_generator(
  
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
  
  )


# 5. Fine-Tuning ----------------------------------------------------------

# Model

model3 <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# UnFreeze

cat('Trainable weights before freezing : ', length(model3$trainable_weights))
unfreeze_weights(conv_base, from = 'block3_conv1')
cat('Trainable weights after freezing : ', length(model3$trainable_weights))

# Fit

history3 <- model3 %>% fit_generator(
  
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
  
  )

# Evaluate

test_generator <- flow_images_from_directory(
  
  test_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
  
  )

model1 %>% 
model2 %>% 
model3 %>% evaluate_generator(test_generator, steps = 50)