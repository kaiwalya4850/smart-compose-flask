import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)

enc_model = keras.models.load_model('encoder-model-final.h5', compile=False)
inf_model = keras.models.load_model('inf-model-final.h5', compile=False)


vocab_max_size = 10000

with open('word_dict-final.json') as f:
    word_dict = json.load(f)
    tokenizer = keras.preprocessing.text.Tokenizer(filters='', num_words=vocab_max_size)
    tokenizer.word_index = word_dict


max_length_in = 21
max_length_out = 20

def tokenize_text(text):
  text = '<start> ' + text.lower() + ' <end>'
  text_tensor = tokenizer.texts_to_sequences([text])
  text_tensor = keras.preprocessing.sequence.pad_sequences(text_tensor, maxlen=max_length_in, padding="post")
  return text_tensor


index_to_word = dict(map(reversed, tokenizer.word_index.items()))

def decode_sequence(input_sentence):
    sentence_tensor = tokenize_text(input_sentence)
    # Encode the input as state vectors.
    state = enc_model.predict(sentence_tensor)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    curr_word = "<start>"
    decoded_sentence = ''

    i = 0
    while curr_word != "<end>" and i < (max_length_out - 1):
        print(target_seq.shape)
        output_tokens, h = inf_model.predict([target_seq, state])

        curr_token = np.argmax(output_tokens[0, 0])

        if (curr_token == 0):
          break;

        curr_word = index_to_word[curr_token]

        decoded_sentence += ' ' + curr_word
        target_seq[0, 0] = curr_token
        state = h
        i += 1

    return decoded_sentence


#ip = input("enter something")

def fin_fxn(ip_txt):
    ip_txt = str(ip_txt)
    texts = []
    texts.append(ip_txt)
    output = list(map(lambda text: (text, decode_sequence(text)), texts))
    output = output[0][1]
    output = output.replace('<end>', '')
    print(output)
    return output


@app.route('/main', methods=["POST", "GET"])
def data_fetch():
    if request.method == "POST":
        text_dat = request.form["smartm"]
        return redirect(url_for("final", to_pass = text_dat))
    else:
        return render_template("main.html") 

@app.route("/<to_pass>")
def final(to_pass):
    ip = str(to_pass)
    f = fin_fxn(ip)
    return f"<h2>Predicted text: {f}</h2>"

if __name__ == "__main__":
    app.run()
