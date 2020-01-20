import os
from pprint import pprint

import scipy.signal
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras_multi_head import MultiHeadAttention

from args import parse_args, channels_val, channels_test, hold_out_test_channels
from compute_text_embeddings import ElmoEmbeddingLayer, BertLayer, \
    create_data_for_finetuning_bert, create_data_for_finetuning_elmo
from evaluation import evaluate
import json
from collections import Counter, OrderedDict
import time
from utils_data_text import get_features_from_data, stemm_list_actions, \
    separate_mapped_visibile_actions, color, compute_predicted_IOU, \
    compute_predicted_IOU_GT, create_data_for_model, get_seqs, compute_median_per_miniclip, method_compare_actions

from keras.layers import Dense, Input, Dropout, Reshape, dot, Embedding, Bidirectional, Flatten, LSTM, Multiply, Add, \
    concatenate
import tensorflow as tf
from keras import backend as K, Sequential
from keras.models import Model

import numpy as np
from keras_self_attention import SeqSelfAttention

# from keras import layers

# # Initialize session
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), allow_soft_placement=True)
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
K.set_session(sess)


def set_random_seed():
    # Set a seed value
    seed_value = 23
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.set_random_seed(seed_value)
    # 6 Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    import torch
    torch.manual_seed(seed_value)


def main_model(input_dim_video):
    max_num_words = 20000
    max_length = 22

    model1 = Sequential()
    model1.add(Embedding(max_num_words, 100, input_length=max_length))
    model1.add(Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model1.add(MultiHeadAttention(head_num=4, name='Multi-Head'))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model1.add(Flatten())
    model1.add(Dense(64))
    model1.summary()

    input2 = Input(shape=(input_dim_video,))
    output2 = Dense(64, activation='relu')(input2)
    model2 = Model(input2, output2)
    model2.summary()
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))

    # MPU
    multiply = Multiply()([model1.output, model2.output])
    add = Add()([model1.output, model2.output])
    concat_multiply_add = concatenate([multiply, add])

    concat = concatenate([model1.output, model2.output])
    FC = Dense(64)(concat)

    concat_all = concatenate([concat_multiply_add, FC])

    output = Dense(64)(concat_all)
    output = Dropout(0.5)(output)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(output)

    merged_model = Model([model1.input, model2.input], main_output)

    # model1.add(Dense(1, activation='sigmoid'))
    merged_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    merged_model.summary()
    return merged_model


# model similar to TALL (alignment score & regression is different + pre-trained model features used)
def create_MPU_model(input_dim_video, input_dim_text, finetune_elmo, finetune_bert):
    # Second input
    if finetune_elmo:
        action_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='action_input')
        action_emb = ElmoEmbeddingLayer()(action_input)
        action_output = Dense(64)(action_emb)

    elif finetune_bert:
        # Build model
        max_seq_length = 256
        in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        # Instantiate the custom Bert Layer defined above
        bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
        action_output = tf.keras.layers.Dense(256, activation="relu")(bert_output)
        action_output = tf.keras.layers.Dense(64)(action_output)

    else:
        print("no finetune")
        action_input = tf.keras.layers.Input(shape=(input_dim_text,), name='action_input')
        action_output = tf.keras.layers.Dense(64)(action_input)

    video_input = tf.keras.layers.Input(shape=(input_dim_video,), dtype='float32', name='video_input')
    # video_input = Input(shape=(input_dim_video[0], input_dim_video[1],), dtype='float32', name='video_input')
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    # video_output = LSTM(64)(video_input)

    video_output = tf.keras.layers.Dense(64)(video_input)

    # MPU
    multiply = tf.keras.layers.Multiply()([action_output, video_output])
    add = tf.keras.layers.Add()([action_output, video_output])
    concat_multiply_add = tf.keras.layers.concatenate([multiply, add])

    concat = tf.keras.layers.concatenate([action_output, video_output])
    FC = tf.keras.layers.Dense(64)(concat)

    concat_all = tf.keras.layers.concatenate([concat_multiply_add, FC])

    output = tf.keras.layers.Dense(64)(concat_all)
    output = tf.keras.layers.Dropout(0.5)(output)

    # And finally we add the main logistic regression layer
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(output)

    if finetune_bert:
        model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, video_input], outputs=[main_output])
    else:
        model = tf.keras.models.Model(inputs=[action_input, video_input], outputs=[main_output])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def create_cosine_sim_model(input_dim_video, input_dim_text):
    action_input = Input(shape=(input_dim_text,), name='action_input')
    action_output = Dense(64)(action_input)
    action_output = Dropout(0.5)(action_output)

    video_input = Input(shape=(input_dim_video,), dtype='float32', name='video_input')
    video_output = Dense(64)(video_input)
    video_output = Dropout(0.5)(video_output)

    # now perform the dot product operation to get a similarity measure
    dot_product = dot([action_output, video_output], axes=1, normalize=True)
    dot_product = Reshape((1,))(dot_product)

    # add the sigmoid output layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(dot_product)

    model = Model(inputs=[action_input, video_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def compute_majority_label_baseline_acc(labels_train, labels_test):
    if Counter(labels_train)[False] > Counter(labels_train)[True]:
        maj_label = False
    else:
        maj_label = True
    maj_labels = [maj_label] * len(labels_test)
    # maj_labels = [True] * len(labels_test)
    nb_correct = 0
    for i in range(len(labels_test)):
        if maj_labels[i] == labels_test[i]:
            nb_correct += 1
    return nb_correct / len(labels_test), maj_labels


def create_main_model(train_data, val_data, test_data, model_name, nb_epochs, balance, config_name):
    print("---------- Running " + config_name + " -------------------")

    [data_clips_train, data_actions_train, labels_train, data_actions_names_train], [data_clips_val, data_actions_val,
                                                                                     labels_val,
                                                                                     data_actions_names_val], \
    [data_clips_test, data_actions_test, labels_test, data_actions_names_test,
     data_clips_names_test] = get_features_from_data(train_data,
                                                     val_data,
                                                     test_data)

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(data_actions_names_train + data_actions_names_val + data_actions_names_test)

    data_actions_train = get_seqs(data_actions_names_train, tokenizer)
    data_actions_val = get_seqs(data_actions_names_val, tokenizer)
    data_actions_test = get_seqs(data_actions_names_test, tokenizer)

    # input_dim_text = len(data_actions_train[0])
    input_dim_video = data_clips_train[0].shape[0]
    print(data_clips_train.shape)

    # print(input_dim_text)
    print(data_actions_train.shape)
    print(input_dim_video)

    model = main_model(input_dim_video)

    config_name = config_name + " video "
    if balance == True:
        file_path_best_model = 'model/Model_params/' + config_name + '.hdf5'
    else:
        file_path_best_model = 'model/model_params_unbalanced/' + config_name + '.hdf5'

    checkpointer = ModelCheckpoint(monitor='val_acc',
                                   filepath=file_path_best_model, verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=20)
    tensorboard = TensorBoard(log_dir="logs/fit/" + time.strftime("%c") + "_" + config_name, histogram_freq=0,
                              write_graph=True)
    callback_list = [earlystopper, checkpointer]

    if not os.path.isfile(file_path_best_model):
        # model.fit(data_actions_train, labels_train,
        #           validation_data=(data_actions_val, labels_val),
        #           epochs=nb_epochs, batch_size=64, verbose=1, callbacks=callback_list)

        model.fit([data_actions_train, data_clips_train], labels_train,
                  validation_data=([data_actions_val, data_clips_val], labels_val),
                  epochs=nb_epochs, batch_size=64, verbose=1, callbacks=callback_list)

    print("Load best model weights from " + file_path_best_model)
    model.load_weights(file_path_best_model)

    # score, acc_train = model.evaluate(data_actions_train, labels_train)
    # score, acc_test = model.evaluate(data_actions_test, labels_test)
    # score, acc_val = model.evaluate(data_actions_val, labels_val)
    # list_predictions = model.predict(data_actions_test)

    score, acc_train = model.evaluate([data_actions_train, data_clips_train], labels_train)
    score, acc_test = model.evaluate([data_actions_test, data_clips_test], labels_test)
    score, acc_val = model.evaluate([data_actions_val, data_clips_val], labels_val)
    list_predictions = model.predict([data_actions_test, data_clips_test])

    print("GT test data: " + str(Counter(labels_test)))
    predicted = list_predictions >= 0.5
    print("Predicted test data: " + str(Counter(x for xs in predicted for x in set(xs))))

    f1_test = f1_score(labels_test, predicted)
    prec_test = precision_score(labels_test, predicted)
    rec_test = recall_score(labels_test, predicted)
    print("precision {0}, recall: {1}, f1: {2}".format(prec_test, rec_test, f1_test))
    print("acc_train: {:0.2f}".format(acc_train))
    print("acc_val: {:0.2f}".format(acc_val))
    print("acc_test: {:0.2f}".format(acc_test))

    return model_name, predicted, list_predictions


def create_model(train_data, val_data, test_data, model_name, nb_epochs, balance, config_name):
    print("---------- Running " + config_name + " -------------------")

    [data_clips_train, data_actions_train, labels_train, data_actions_names_train], [data_clips_val, data_actions_val,
                                                                                     labels_val,
                                                                                     data_actions_names_val], \
    [data_clips_test, data_actions_test, labels_test, data_actions_names_test,
     data_clips_names_test] = get_features_from_data(train_data,
                                                     val_data,
                                                     test_data)
    # input_dim_text = data_actions_val[0].shape[0]
    input_dim_text = len(data_actions_train[0])
    input_dim_video = data_clips_train[0].shape[0]
    # input_dim_video = data_clips_train[0].shape

    if config_name.split(" + ")[2] == "finetuned ELMo":
        data_actions_train, data_actions_val, data_actions_test, data_clips_train, data_clips_val, data_clips_test = create_data_for_finetuning_elmo(
            data_actions_names_train, data_actions_names_val, data_actions_names_test,
            data_clips_train, data_clips_val, data_clips_test)
        finetune_elmo = True
    else:
        finetune_elmo = False

    if config_name.split(" + ")[2] == "finetuned Bert":
        [data_actions_train, data_actions_val, data_actions_test], [data_clips_train, data_clips_val, data_clips_test], \
        [train_input_ids, val_input_ids, test_input_ids], [train_input_masks, val_input_masks, test_input_masks], \
        [train_segment_ids, val_segment_ids, test_segment_ids] = \
            create_data_for_finetuning_bert(data_actions_names_train, data_actions_names_val, data_actions_names_test,
                                            data_clips_train, data_clips_val, data_clips_test, labels_train, labels_val,
                                            labels_test)

        finetune_bert = True
    else:
        finetune_bert = False

    if model_name == "MPU":
        model = create_MPU_model(input_dim_video, input_dim_text, finetune_elmo, finetune_bert)
    elif model_name == "cosine sim":
        model = create_cosine_sim_model(input_dim_video, input_dim_text)
    else:
        raise ValueError("Wrong model name!")
    if balance == True:
        file_path_best_model = 'model/Model_params/' + config_name + '.hdf5'
    else:
        file_path_best_model = 'model/model_params_unbalanced/' + config_name + '.hdf5'

    checkpointer = ModelCheckpoint(monitor='val_acc',
                                   filepath=file_path_best_model,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=10)
    tensorboard = TensorBoard(log_dir="logs/fit/" + time.strftime("%c") + "_" + config_name, histogram_freq=0,
                              write_graph=True)
    callback_list = [earlystopper, checkpointer]

    if not os.path.isfile(file_path_best_model):

        if finetune_bert:
            session = tf.keras.backend.get_session()
            init = tf.global_variables_initializer()
            session.run(init)
            model.fit([train_input_ids, train_input_masks, train_segment_ids, data_clips_train], labels_train,
                      validation_data=([val_input_ids, val_input_masks, val_segment_ids, data_clips_val], labels_val),
                      epochs=nb_epochs, batch_size=64, verbose=1, callbacks=callback_list)
        else:
            model.fit([data_actions_train, data_clips_train], labels_train,
                      validation_data=([data_actions_val, data_clips_val], labels_val),
                      epochs=nb_epochs, batch_size=64, verbose=1, callbacks=callback_list)

    print("Load best model weights from " + file_path_best_model)
    model.load_weights(file_path_best_model)

    if finetune_bert:
        score, acc_train = model.evaluate([train_input_ids, train_input_masks, train_segment_ids, data_clips_train],
                                          labels_train)
        score, acc_test = model.evaluate([test_input_ids, test_input_masks, test_segment_ids, data_clips_test],
                                         labels_test)
        score, acc_val = model.evaluate([val_input_ids, val_input_masks, val_segment_ids, data_clips_val], labels_val)

        list_predictions = model.predict([test_input_ids, test_input_masks, test_segment_ids, data_clips_test])

    else:
        score, acc_train = model.evaluate([data_actions_train, data_clips_train], labels_train)
        score, acc_test = model.evaluate([data_actions_test, data_clips_test], labels_test)
        score, acc_val = model.evaluate([data_actions_val, data_clips_val], labels_val)
        list_predictions = model.predict([data_actions_test, data_clips_test])

    print("GT test data: " + str(Counter(labels_test)))
    # predicted = list_predictions >= 0
    predicted = list_predictions >= 0.5

    ## not ok -- acc high, but low f1 -> worse results
    # predicted = compute_median_per_miniclip(data_actions_names_test, data_clips_names_test, predicted, labels_test)
    # print("Predicted test data: " + str(Counter(predicted)))

    print("Predicted test data: " + str(Counter(x for xs in predicted for x in set(xs))))


    f1_test = f1_score(labels_test, predicted)
    prec_test = precision_score(labels_test, predicted)
    rec_test = recall_score(labels_test, predicted)
    print("precision {0}, recall: {1}, f1: {2}".format(prec_test, rec_test, f1_test))
    print("acc_train: {:0.2f}".format(acc_train))
    print("acc_val: {:0.2f}".format(acc_val))
    print("acc_test: {:0.2f}".format(acc_test))

    return model_name, predicted, list_predictions


def test_alignment():
    with open("data/mapped_actions_time_label.json") as f:
        actions_time_label = json.loads(f.read())

    miniclips_list_stemmed_visibile_actions = {}

    for channel in ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1", "6p0", "6p1", "7p0", "7p1",
                    "8p0", "8p1", "9p0", "9p1", "10p0", "10p1"]:
        # for channel in ["1p0", "1p1", "5p0", "5p1"]:
        # for channel in ["1p0"]:

        # extract the visible ones and stem them
        list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions, list_time_visibile, list_time_not_visibile = separate_mapped_visibile_actions(
            actions_time_label, channel)
        list_stemmed_visibile_actions = stemm_list_actions(visible_actions, "data/dict_action_pos_concreteness.json")

        for index in range(len(list_stemmed_visibile_actions)):
            miniclip = list_miniclips_visibile[index]
            action = list_stemmed_visibile_actions[index]
            time = list_time_visibile[index]
            miniclip_action = miniclip + ", " + action
            miniclips_list_stemmed_visibile_actions[miniclip_action] = [time]

    with open("data/results/dict_predicted_alignment.json", 'w+') as fp:
        json.dump(miniclips_list_stemmed_visibile_actions, fp)


def create_config_name(args):
    if args.model_name == "alignment":
        config_name = "alignment"
    elif args.model_name == "system max":
        config_name = "system max"
    else:
        if args.finetune:
            config_name = args.clip_length + " + " + args.model_name + " + " + "finetuned " + args.type_action_emb + " + " + str(
                args.epochs)
            print("FINETUNING! " + config_name)
        else:
            config_name = args.clip_length + " + " + args.model_name + " + " + args.type_action_emb + " + " + str(
                args.epochs)
        if args.add_cluster:
            print("Add cluster info")
            config_name = config_name + " + cluster"

    return config_name


def main():
    set_random_seed()
    args = parse_args()
    config_name = create_config_name(args)

    if config_name == "alignment":
        test_alignment()
        for channel_test in channels_test:
            evaluate(config_name, channel_test)
    else:
        '''
            Create data
        '''
        train_data, val_data, test_data = \
            create_data_for_model(args.type_action_emb, args.balance, args.add_cluster,
                                  path_all_annotations="data/dict_all_annotations" + args.clip_length + ".json",
                                  path_I3D_features="../i3d_keras/data/results_features_overlapping_" + args.clip_length + "/",
                                  channels_val=channels_val,
                                  channels_test=channels_test,
                                  hold_out_test_channels=hold_out_test_channels)

        if config_name == "system max":
            compute_predicted_IOU_GT(test_data, args.clip_length)
            for channel_test in channels_test:
                evaluate("GT", channel_test)
        else:
            '''
                    Create model
            # '''
            # model_name, predicted, list_predictions = create_model(train_data, val_data, test_data, args.model_name,
            #                                                        args.epochs,
            #                                                        args.balance, config_name)


            # model_name, predicted, list_predictions = create_main_model(train_data, val_data, test_data, "Main",
            #                                                             args.epochs,
            #                                                             args.balance, config_name)

            predicted, list_predictions = method_compare_actions(train_data, val_data, test_data)
            config_name = "compare actions 17"
            '''
                Majority (actions are visible in all clips)
            '''
            # [_, _, labels_train, _], [_, _, labels_val, _], [_, _, labels_test, _] = get_features_from_data(train_data,
            #                                                                                                 val_data,
            #                                                                                                 test_data)
            # maj_val, maj_labels = compute_majority_label_baseline_acc(labels_train, labels_val)
            # maj_test, maj_labels = compute_majority_label_baseline_acc(labels_train, labels_test)
            # print("maj_val: {:0.2f}".format(maj_val))
            # print("maj_test: {:0.2f}".format(maj_test))

            '''
                    Evaluate
            '''
            compute_predicted_IOU(config_name, predicted, test_data, args.clip_length, list_predictions)
            for channel_test in channels_test:
                evaluate(config_name, channel_test)


if __name__ == "__main__":
    main()
