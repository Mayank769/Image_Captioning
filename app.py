import tensorflow as tf
import cloudinary.uploader
import cloudinary
from pickle import load
from flask import Flask, request, jsonify, render_template,flash
import os
from Image_captioning_Model import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    checkpoint_path = "./checkpoints/train/ckpt-1"
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    # here restoring the checkpoint
    ckpt.restore(checkpoint_path)
    train_captions = load(open('./captions.pkl', 'rb'))
    # Find the maximum length of any caption in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)
    # Choose the top 5000 words from the vocabulary 
    top_k = 5000
    #limit the vocabulary size to the top 5,000 words (to save memory) and replace all other words with the token "UNK" (unknown).
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    #Updates internal vocabulary based on a list of texts.
    tokenizer.fit_on_texts(train_captions)
    #pad all sequences to be the same length as the longest one.
    tokenizer.word_index['<pad>'] = 0  #just making the index of word "<pad>" to 0
    tokenizer.index_word[0] = '<pad>'# making 0 index to "<pad>"
    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length=calc_max_length(cap_vector)
    
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file=request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    img=file.read()
    image_path = cloudinary.uploader.upload(img)
    result1, _ = evaluate_standard(img,encoder,decoder,tokenizer,max_length)
    result2, _ = evaluate_greedy(img,encoder,decoder,tokenizer,max_length)
    
    result1=modify(result1)
    result2=modify(result2)
    captn1=' '.join(result1)
    captn2=' '.join(result2)
    return render_template('Caption_page.html',caption1=captn1,caption2=captn2,Image=image_path['url'])

if __name__ == '__main__':
    app.run(debug=True)
#,port=80,host='0.0.0.0'