import json, codecs

import torch
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.tokenization_auto import AutoTokenizer
from transformers.modeling_bert import BertModel
from nltk import word_tokenize
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

embeddings_index = dict()
with open("data/glove.6B.50d.txt") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

dimension_embedding = len(embeddings_index.get("example"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def avg_GLoVe_action_emb(action):
    # no prev or next action: ned to distinguish between cases when action is not recognized
    if action == "":
        average_word_embedding = np.ones((1, dimension_embedding), dtype='float32') * 10
    else:
        list_words = word_tokenize(action)
        set_words_not_in_glove = set()
        nb_words = 0
        average_word_embedding = np.zeros((1, dimension_embedding), dtype='float32')
        for word in list_words:
            if word in set_words_not_in_glove:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                set_words_not_in_glove.add(word)
                continue
            word_embedding = np.asarray(embedding_vector)
            average_word_embedding += word_embedding
            nb_words += 1
        if nb_words != 0:
            average_word_embedding = average_word_embedding / nb_words

        if (average_word_embedding == np.zeros((1,), dtype=np.float32)).all():
            # couldn't find any word of the action in the vocabulary -> initialize random
            average_word_embedding = np.random.rand(1, dimension_embedding).astype('float32')

    return average_word_embedding.reshape(50)


def embed_elmo2():
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


def create_elmo_embddings(list_all_actions):
    dict_action_embeddings = {}
    start = time.time()
    embed_fn = embed_elmo2()
    end = time.time()
    print("Load ELMo model took " + str(end - start))
    print("Running ELMo ... ")
    for action in tqdm(list_all_actions):
        # emb_action = embed_fn([action]).reshape(1024)
        emb_action = embed_fn([action]).reshape(-1)
        dict_action_embeddings[action] = emb_action

    with open('data/dict_action_embeddings_ELMo.json', 'w+') as outfile:
        json.dump(dict_action_embeddings, outfile, cls=NumpyEncoder)

    return dict_action_embeddings


def create_glove_embeddings(list_all_actions):
    dict_action_embeddings = {}
    for action in list_all_actions:
        emb_action = avg_GLoVe_action_emb(action)
        dict_action_embeddings[action] = emb_action
    return dict_action_embeddings


def create_bert_embeddings(list_all_actions):
    start = time.time()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model (weights)
    pretrained_model_name = 'data/epoch_29/'
    model = BertModel.from_pretrained(pretrained_model_name)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    end = time.time()
    print("Load BERT model took " + str(end - start))
    dict_action_embeddings = {}
    print("Running BERT ... ")
    for action in tqdm(list_all_actions):
        emb_action = get_bert_finetuned_embeddings(model, tokenizer, action)
        # emb_action = finetune_bert(model,tokenizer, action)
        dict_action_embeddings[action] = emb_action.reshape(-1)

    with open('data/dict_action_embeddings_Bert2.json', 'w+') as outfile:
        json.dump(dict_action_embeddings, outfile, cls=NumpyEncoder)
    return dict_action_embeddings

# def finetune_bert(model,tokenizer, action):
#     action = "[CLS] " + action + " [SEP]"
#     tokenized_texts = [tokenizer.tokenize(action)]
#     MAX_LEN = 128
#     # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
#     input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
#     # Pad our input tokens
#     input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
#
#     # Create attention masks
#     attention_masks = []
#
#     # Create a mask of 1s for each token followed by 0s for padding
#     for seq in input_ids:
#         seq_mask = [float(i > 0) for i in seq]
#         attention_masks.append(seq_mask)
#
#     with torch.no_grad():
#         # Forward pass, calculate logit predictions
#         model_output = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
#         last_hidden_states = model_output[0]  # The last hidden-state is the first element of the output tuple
#
#         mean_sentence_embedding = torch.mean(last_hidden_states, 1)  # average token embeddings
#         sentence_embedding = torch.Tensor.numpy(mean_sentence_embedding)  # average token embeddings
#         return sentence_embedding

#domain adaptation
def get_bert_finetuned_embeddings(model, tokenizer, action):

    marked_text = "[CLS] " + action + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict hidden states features for each layer
    with torch.no_grad():
        model_output = model(tokens_tensor, segments_tensors)
        last_hidden_states = model_output[0]  # The last hidden-state is the first element of the output tuple

    mean_sentence_embedding = torch.mean(last_hidden_states, 1)  # average token embeddings
    sentence_embedding = torch.Tensor.numpy(mean_sentence_embedding)  # average token embeddings
    return sentence_embedding


def main():
    get_bert_finetuned_embeddings("lala")


if __name__ == '__main__':
    main()
