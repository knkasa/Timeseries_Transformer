# Example of timeseries Transformer. 
# https://keras.io/examples/timeseries/timeseries_transformer_classification/ 

import os
import numpy as np
import tensorflow as tf

os.chdir('C:\my_working_env\deeplearning_practice')

#------------------------------------------------------------------

def readucr(filename):
    data = np.loadtxt( filename, delimiter="\t" )
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# Reshape to make it NxTxD (N=# of samples), T=lookup time, D=# of columns.
# Note number of lookup is 500.  This is much more than lstm could handle.
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

# Randomly shuffle the data.
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Make y binary 0 or 1.  
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

#----------------- Setup neural network. -------------------------------------
        
class transformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        head_size = 128
        num_heads = 4
        num_filter = 4
        dropout = 0.2
                
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Note attention layer requires dimension to be NxTxD just like LSTM.
        self.mha = tf.keras.layers.MultiHeadAttention(
                                                    key_dim=head_size, 
                                                    num_heads=num_heads, 
                                                    dropout=dropout,
                                                    )

        self.dropout = tf.keras.layers.Dropout(dropout)
        
        self.conv1D = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=1, activation="relu",)   
        self.conv1D_2 = tf.keras.layers.Conv1D(filters=1, kernel_size=1) # The dimension needs to be NxTx1.  
    
    def call(self, x):
        inputs = x
        x = self.layer_norm(x)
        x = self.mha(x,x)
        x = self.dropout(x)
        res = x + inputs

        x = self.layer_norm(res)
        x = self.conv1D(x)
        x = self.dropout(x)
        x = self.conv1D_2(x)
        return x + res
        
class transformer_model(tf.keras.Model):
    def __init__(self, input_shape:int ):
        super().__init__()
        
        self.num_block_layers = 4
        self.num_transformer = 2
        dense_units = 64
        dropout = 0.2
        
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
        
        self.dense_layer = tf.keras.layers.Dense(dense_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.transformer_layer = transformer()
        self.output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        
        # Note that after the transformer_layer, the input dimension stays the same.  
        for _ in range(self.num_transformer):  # stack number of transformer layers.
            x = self.transformer_layer(x)
           
        x = self.pooling(x)  # Need to reduce dimension from NxTxD to NxT
        x = self.dense_layer(x)
        x = self.dropout(x)
        outputs = self.output_layer(x)
        
        return outputs


input_shape = x_train.shape[1]
model = transformer_model(input_shape)  
model.build( input_shape=(None,input_shape,1) )    # Define input shape here NxD D=# of columns.  For LSTM, NxTxD (None, T, D).  

#--------- Train the model ----------------------------------------------

model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["sparse_categorical_accuracy"],
            )

callback_opt = [tf.keras.callbacks.EarlyStopping(patience=10, 
                                           restore_best_weights=True)]

model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=1,
        batch_size=64,
        callbacks=callback_opt,
        )
        
model.predict( x_test[0:2] )
model.evaluate(x_test, y_test, verbose=1)





    