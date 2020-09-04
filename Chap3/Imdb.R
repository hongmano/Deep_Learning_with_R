
# 1. Packages / Options ---------------------------------------------------

if (!require(keras)) install.packages('keras'); require(keras)

# 2. Data Loading ---------------------------------------------------------

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

# 3. Data Wrangling -------------------------------------------------------

# train_data 내용 확인

decoded_review <- sapply(train_data[[1]], function(index){
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else '?'
})

# 시퀀스를 Binary Matrix로

vectorize_sequences <- function(sequences, dimension = 10000){
  
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)){
    results[i, sequences[[i]]] <- 1
  }
  results
}

# 하나의 시퀀스에 해당 단어가 등장하면 1, 아니면 0

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# 4. Model -----------------------------------------------------------------

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  
  optimizer = optimizer_rmsprop(lr = .001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')

  )


# 5. Hold-Out -------------------------------------------------------------

x_val <- x_train[1:10000, ]
x_train <- x_train[-c(1:10000), ]
y_val <- y_train[1:10000]
y_train <- y_train[-c(1:10000)]

# 6. Fit ------------------------------------------------------------------

history <- model %>% fit(
  x_train,
  y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

str(history)
plot(history)
