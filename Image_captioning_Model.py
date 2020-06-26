import matplotlib.pyplot as plt
import tensorflow as tf
import sys, time, os, warnings
import pickle
import time
import numpy as np

#Preprocessing for inceptionV3 model
def load_image(img):
    #img = tf.io.read_file(image_path)   #reads the file mentioned in the path
    img = tf.image.decode_jpeg(img, channels=3)  #convert the compressed string to a 3D uint8 tensor
    img = tf.image.resize(img, (299, 299))  #resizes image to desired size
    img = tf.keras.applications.inception_v3.preprocess_input(img) #preprocess the image to make it suitable as input for inception model(normalized btw -1 & 1)
    return img


#initialize our model weights to imagenet model weights(1.5 million images and 1000 classes) 
#not including the classification layer 
image_model = tf.keras.applications.InceptionV3(include_top=False)
new_input = image_model.input   # input layer
hidden_layer = image_model.layers[-1].output  # hidden layers

#creates a model with input layer=>new_input and output layers=>hidden_layer
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)



BATCH_SIZE = 64  # this is batch size which we r going to send in the model
BUFFER_SIZE = 1000 # i donno about this
embedding_dim = 256 # we r getting output from inceptionV3 model of dimension 64*2048..that we will convert into 64*embedding_dim
units = 512 #number of unit in the rnn model
vocab_size = 5000+ 1 # this u know 1 added bcoz of <unk>

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048 #features shape ..remember we r getting output of 8*8*2048
attention_features_shape = 64 #this we decided on our own


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__() #calling / initializing  its super class
        self.W1 = tf.keras.layers.Dense(units)    #making layer object which gives output of dimension (batch_size,units)
        self.W2 = tf.keras.layers.Dense(units)    
        self.V = tf.keras.layers.Dense(1)   

    def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim) so this one is features of image

    # hidden shape == (batch_size, hidden_size)  ...this one is previous hidden state
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1) # expanding dim

    # score shape == (batch_size, 64, hidden_size) (this is eij)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))#  concatenate previous hidden state and features
    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1) #this is alpha

    # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features         #(batch_size,64,hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1) #this is final context vector

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #converts it into dense vector of size (vocab_size,input_length,embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)  #output and hidden state

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def evaluate_standard(image,encoder,decoder,tokenizer,max_length):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    #creating image tensor(by extracting features) so that we can send it as input into encoder
    temp_input = tf.expand_dims(load_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def evaluate_greedy(image,encoder,decoder,tokenizer,max_length):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    #creating image tensor(by extracting features) so that we can send it as input into encoder
    temp_input = tf.expand_dims(load_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
    
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def modify(result):
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass
    return result