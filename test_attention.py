import keras
import numpy as np
from keras_transformer import get_model, get_encoders
import numpy as np
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward
from keras_pos_embd import TrigPosEmbedding
from keras_embed_sim import EmbeddingRet, EmbeddingSim

import tensorflow as tf
import tensorflow_hub as hub

# Build a small toy token dictionary
token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }

def vanilla_transformer(encoder_inputs, decoder_inputs, decoder_outputs):
    # Build the model
    model = get_model(
        token_num=len(token_dict),
        embed_dim=30,
        encoder_num=3,
        decoder_num=2,
        head_num=3,
        hidden_dim=120,
        attention_activation='relu',
        feed_forward_activation='relu',
        dropout_rate=0.05,
        embed_weights=np.random.random((13, 30)),
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
    )
    model.summary()

    # Train the model
    print(np.asarray(encoder_inputs * 1000).shape)
    print(np.asarray(decoder_inputs * 1000).shape)
    print(np.asarray(decoder_outputs * 1000).shape)
    model.fit(
        x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
        y=np.asarray(decoder_outputs * 1000),
        epochs=5,
    )



'''
A sentence is first split into words
'''

def create_word_level_sentence_embedd(sentence):
    tokens = sentence.split()
    for token in tokens:
        if token not in token_dict:
            token_dict[token] = len(token_dict)

    # Generate toy data
    encoder_inputs_no_padding = []
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
    for i in range(1, len(tokens) - 1):
        encode_tokens, decode_tokens = tokens[:i], tokens[i:]
        encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(encode_tokens))
        output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
        decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
        encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
        decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
        output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
        encoder_inputs_no_padding.append(encode_tokens[:i + 2])
        encoder_inputs.append(encode_tokens)
        encoder_inputs.append(decode_tokens)
        decoder_outputs.append(output_tokens)

    return  encoder_inputs, decoder_inputs, decoder_outputs

# https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/
def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)),
    	signature="default", as_dict=True)["default"]

def model_language_encoder():
    input_text = keras.layers.Input(shape=(1,), dtype=tf.string)
    embedding = keras.layers.Lambda(UniversalEmbedding,
                              output_shape=(512,))(input_text)

    # dense = layers.Dense(256, activation='relu')(embedding)
    # pred = layers.Dense(category_counts, activation='softmax')(dense)
    # model = Model(inputs=[input_text], outputs=pred)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])


def main():
   # encoder_inputs, _, _ = create_word_level_sentence_embedd("oana is shopping")
    #language_encoder(encoder_inputs)
   module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
   # Import the Universal Sentence Encoder's TF Hub module
   embed = hub.Module(module_url)
   session = tf.Session()
   session.run([tf.global_variables_initializer(), tf.tables_initializer()])

   sentences = ["oana is shopping"]
   sentence_embeddings = session.run(embed(sentences))
   print(sentence_embeddings)


if __name__ == "__main__":
    main()
