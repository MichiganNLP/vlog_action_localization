import json, codecs

import torch
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedModel
from transformers.tokenization_auto import AutoTokenizer
from transformers.modeling_bert import BertModel
from nltk import word_tokenize
import numpy as np
import time

from keras import backend as K, Model
from keras.engine import Layer
from keras import layers
import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer

# # Initialize session
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), allow_soft_placement=True)
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
K.set_session(sess)

embeddings_index = dict()
with open("data/glove.6B.50d.txt") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

dimension_embedding = len(embeddings_index.get("example"))


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']

        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions

def build_elmo_model():
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    dropout = layers.Dropout(0.5)(dense)
    pred = layers.Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=[input_text], outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# With the default signature, the module takes untokenized sentences as input
# The input tensor is a string tensor with shape [batch_size].
# The module tokenizes each string by splitting on spaces
def test_elmo2():
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(
        ["the cat", "dogs are big"],
        signature="default",
        as_dict=True)["word_emb"]
    # convert from Tensor to numpy array
    array = K.eval(embeddings)
    return array


'''
The output dictionary contains:

word_emb: the character-based word representations with shape [batch_size, max_length, 512].
lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
elmo: the weighted sum of the 3 layers, where the weights are trainable. 
This tensor has shape [batch_size, max_length, 1024]
default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
'''


def load_elmo_embedding(train_data):
    print("# ---- loading elmo embeddings ---")
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(
        train_data,
        signature="default",
        as_dict=True)["default"] #["word_emb"]
    # K.eval is computationally expensive (might be doing some convs)
    #array = K.eval(embeddings)
    #print("# ---- loaded elmo embeddings ---")
    return K.eval(embeddings)


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)




def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label




def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples




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

def create_elmo_embedding(action):
    start = time.time()
    embed_fn = embed_elmo2()
    end = time.time()
    print("Load ELMo model took " + str(end - start))
    print("Running ELMo ... ")
    emb_action = embed_fn([action]).reshape(-1)
    return emb_action


def save_elmo_embddings(list_all_actions):
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

    # with open('data/dict_action_embeddings_ELMo_vb_particle.json', 'w+') as outfile:
    with open('data/embeddings/dict_action_embeddings_COIN.json', 'w+') as outfile:
        json.dump(dict_action_embeddings, outfile, cls=NumpyEncoder)

    return dict_action_embeddings


def create_glove_embeddings(list_all_actions):
    dict_action_embeddings = {}
    for action in list_all_actions:
        emb_action = avg_GLoVe_action_emb(action)
        dict_action_embeddings[action] = emb_action
    return dict_action_embeddings


def create_bert_embeddings(list_all_actions):
    tokenizer_name = 'bert-base-uncased'

    # pretrained_model_name = 'data/epoch_29/'
    pretrained_model_name = tokenizer_name

    start = time.time()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Load pre-trained model (weights)
    # model = PreTrainedModel.from_pretrained(pretrained_model_name)
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

    with open('data/embeddings/dict_action_embeddings_Bert_Charades.json', 'w+') as outfile:
        json.dump(dict_action_embeddings, outfile, cls=NumpyEncoder)
    return dict_action_embeddings


def create_data_for_finetuning_bert(data_actions_names_train, data_actions_names_val, data_actions_names_test,
                                    data_clips_train, data_clips_val, data_clips_test, labels_train, labels_val,
                                    labels_test):
    max_seq_length = 256
    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")

    data_actions_train = [" ".join(t.split()[0:max_seq_length]) for t in data_actions_names_train]
    data_actions_train = np.array(data_actions_train, dtype=object)[:, np.newaxis]

    data_actions_val = [" ".join(t.split()[0:max_seq_length]) for t in data_actions_names_val]
    data_actions_val = np.array(data_actions_val, dtype=object)[:, np.newaxis]

    data_actions_test = [" ".join(t.split()[0:max_seq_length]) for t in data_actions_names_test]
    data_actions_test = np.array(data_actions_test, dtype=object)[:, np.newaxis]

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(data_actions_train, labels_train)
    val_examples = convert_text_to_examples(data_actions_val, labels_val)
    test_examples = convert_text_to_examples(data_actions_test, labels_test)

    # Convert to features
    (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_seq_length
    )
    (
        val_input_ids,
        val_input_masks,
        val_segment_ids,
        val_labels,
    ) = convert_examples_to_features(
        tokenizer, val_examples, max_seq_length=max_seq_length
    )
    (
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels,
    ) = convert_examples_to_features(
        tokenizer, test_examples, max_seq_length=max_seq_length
    )

    data_clips_train = np.array(data_clips_train, dtype=object)
    data_clips_val = np.array(data_clips_val, dtype=object)
    data_clips_test = np.array(data_clips_test, dtype=object)

    return [data_actions_train, data_actions_val, data_actions_test], [data_clips_train, data_clips_val, data_clips_test], \
           [train_input_ids, val_input_ids, test_input_ids], [train_input_masks, val_input_masks, test_input_masks], \
           [train_segment_ids, val_segment_ids, test_segment_ids]


def create_data_for_finetuning_elmo(data_actions_names_train, data_actions_names_val, data_actions_names_test,
                                    data_clips_train, data_clips_val, data_clips_test):
    print("before data_clips_train len: {0}".format(data_clips_train[0].shape[0]))
    data_actions_train = np.array(data_actions_names_train, dtype=object)[:, np.newaxis]
    data_actions_val = np.array(data_actions_names_val, dtype=object)[:, np.newaxis]
    data_actions_test = np.array(data_actions_names_test, dtype=object)[:, np.newaxis]

    data_clips_train = np.array(data_clips_train, dtype=object)
    data_clips_val = np.array(data_clips_val, dtype=object)
    data_clips_test = np.array(data_clips_test, dtype=object)
    print("after data_clips_train.shape: {0}".format(data_clips_train.shape))
    print("Elmo actions, data_actions_train.shape: {0}".format(data_actions_train.shape))

    return data_actions_train, data_actions_val, data_actions_test, data_clips_train, data_clips_val, data_clips_test



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

    # [CLS] token representation would be: sentence_embedding = last_hidden_states[:, 0, :]

    sentence_embedding = torch.mean(last_hidden_states, 1)  # average token embeddings

    return sentence_embedding.cpu().numpy()


def main():
    get_bert_finetuned_embeddings("lala")


if __name__ == '__main__':
    main()
