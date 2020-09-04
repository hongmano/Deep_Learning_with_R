
# 1. Packages / Options ---------------------------------------------------

if (!require(keras)) install.packages('keras'); require(keras)
if (!require(ggplot2)) install.packages('ggplot2'); require(ggplot2)

# 2. Data Loading ---------------------------------------------------------

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

# 3. Data Wrangling -------------------------------------------------------

train_data <- scale(train_data)
test_data <- scale(test_data)

# 4. Model ----------------------------------------------------------------

build_model <- function(){
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = 'relu', input_shape = dim(train_data)[2]) %>% 
    layer_dense(units = 64, activation = 'relu') %>% 
    layer_dense(units = 1)
  
  model %>% compile(
    
    optimizer = 'rmsprop',
    loss = 'mse',
    metrics = c('mae')
    
  )
}

# 5. K-Fold Validation ----------------------------------------------------

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = F)

num_epochs <- 500
all_mae_histories <- NULL

for (i in 1:k){
  
  cat('processing fold #', i, '\n')
  
  val_indices <- which(folds == i, arr.ind = T)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(
    
    partial_train_data,
    partial_train_targets,
    validation_split = 1,
    epochs = num_epochs,
    validation_data = list(val_data, val_targets),
    batch_size = 1,
    verbose = 0
    
  )
  
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}


average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

average_mae_history %>% 
  ggplot(aes(x = epoch,
             y = validation_mae)) + 
  geom_smooth(method = 'loess',
              formula = y ~ x)

              