
# 1. Packages / Options ---------------------------------------------------

if (!require(keras)) install.packages('keras'); require(keras)

# 2. Data Loading ---------------------------------------------------------

original_dataset_dir <- 'C:/Users/Mano/Desktop/dogs-vs-cats/train/train'

base_dir <- 'C:/Users/Mano/Desktop/dogs-vs-cats/small'
dir.create(base_dir)

train_dir <- file.path(base_dir, 'train')
dir.create(train_dir)
validation_dir <- file.path(base_dir, 'validation')
dir.create(validation_dir)
test_dir <- file.path(base_dir, 'test')
dir.create(test_dir)

train_cats_dir <- file.path(train_dir, 'cats')
dir.create(train_cats_dir)

train_dogs_dir <- file.path(train_dir, 'dogs')
dir.create(train_dogs_dir)

validation_cats_dir <- file.path(validation_dir, 'cats')
dir.create(validation_cats_dir)

validation_dogs_dir <- file.path(validation_dir, 'dogs')
dir.create(validation_dogs_dir)

test_cats_dir <- file.path(test_dir, 'cats')
dir.create(test_cats_dir)

test_dogs_dir <- file.path(test_dir, 'dogs')
dir.create(test_dogs_dir)

fnames <- paste0('cat.', 1:1000, '.jpg')
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_cats_dir))

fnames <- paste0('cat.', 1001:1500, '.jpg')
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_cats_dir))

fnames <- paste0('cat.', 1501:2000, '.jpg')
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_cats_dir))

fnames <- paste0('dog.', 1:1000, '.jpg')
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_dogs_dir))

fnames <- paste0('dog.', 1001:1500, '.jpg')
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_dogs_dir))

fnames <- paste0('dog.', 1501:2000, '.jpg')
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_dogs_dir))

# 3. Data Augmentation ----------------------------------------------------

datagen <- image_data_generator(
  
  rescale = 1/255,
  rotation_range = 40, # 회전
  width_shift_range = .2, # 가로 변환
  height_shift_range = .2, # 세로 변환
  shear_range = .2, # 전단 변환
  zoom_range = .2, # 확대 변환
  horizontal_flip = T, # 좌우 반전
  fill_mode = 'nearest'
  
  )

augmentation_generator <- flow_images_from_data(
  
  img_array,
  generator = datagen,
  batch_size = 1
  
  )

# 3. Utils ----------------------------------------------------------------

validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
  
)

validation_generator <- flow_images_from_directory(
  
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = 'binary'
  
)

# 4. Model ----------------------------------------------------------------

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  
  loss = loss_binary_crossentropy,
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = metric_binary_accuracy
  
)

# 5. Fit ------------------------------------------------------------------

histroy <- model %>% fit_generator(
  
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
  
  )

model %>% save_model_hdf5('cats_and_dogs_small.h5')
