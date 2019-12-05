import os
import random
from _operator import add

import scipy.signal
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from compute_text_embeddings import create_glove_embeddings, create_bert_embeddings, \
    ElmoEmbeddingLayer, save_elmo_embddings, BertLayer, create_tokenizer_from_hub_module, convert_text_to_examples, \
    convert_examples_to_features, NumpyEncoder
from evaluation import evaluate
import json
import numpy as np
from collections import Counter
from tabulate import tabulate
import time
from utils_data_text import group, get_features_from_data, stemm_list_actions, \
    separate_mapped_visibile_actions, color, compute_action, add_cluster_data

from keras.layers import Dense, Input, Multiply, Add, concatenate, Dropout, Reshape, dot, LSTM
from utils_data_video import load_data_from_I3D, average_i3d_features
import argparse

import tensorflow as tf
from keras import backend as K, Model

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


# # model similar to TALL (alignment score & regression is different + pre-trained model features used)
# def create_MPU_model(input_dim_video, input_dim_text):
#     action_input = Input(shape=(input_dim_text,), name='action_input')
#     action_output = Dense(64)(action_input)
#
#     video_input = Input(shape=(input_dim_video,), dtype='float32', name='video_input')
#     video_output = Dense(64)(video_input)
#
#     # MPU
#     multiply = Multiply()([action_output, video_output])
#     add = Add()([action_output, video_output])
#     concat = concatenate([multiply, add])
#     concat = concatenate([concat, action_output])
#     concat = concatenate([concat, video_output])
#
#     output = Dense(64, activation='relu')(concat)
#     output = Dropout(0.5)(output)
#
#     # And finally we add the main logistic regression layer
#     main_output = Dense(1, activation='sigmoid', name='main_output')(output)
#
#     model = Model(inputs=[video_input, action_input], outputs=[main_output])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     model.summary()
#
#     return model


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


def load_text_embeddings(type_action_emb, dict_all_annotations, all_actions, use_nouns, use_particle):
    set_actions = set()
    for clip in list(dict_all_annotations.keys()):
        list_action_label = dict_all_annotations[clip]
        for [action, _] in list_action_label:
            if not all_actions:
                action, _ = compute_action(action, use_nouns, use_particle)
            set_actions.add(action)
    list_all_actions = list(set_actions)

    if type_action_emb == "GloVe":
        return create_glove_embeddings(list_all_actions)
    elif type_action_emb == "ELMo":
        with open('data/embeddings/dict_action_embeddings_ELMo.json') as f:
            # with open('data/embeddings/dict_action_embeddings_ELMo_vb_particle.json') as f:
            json_load = json.loads(f.read())
        return json_load
        #return save_elmo_embddings(list_all_actions)  # if need to create new
    elif type_action_emb == "Bert":
        with open('data/embeddings/dict_action_embeddings_Bert2.json') as f:
            json_load = json.loads(f.read())
        return json_load
        #return create_bert_embeddings(list_all_actions)
    elif type_action_emb == "DNT":
        with open('steve_human_action/dict_action_emb_DNT.json') as f:
            json_load = json.loads(f.read())
        return json_load
    else:
        raise ValueError("Wrong action emb type")


def split_data_train_val_test(dict_all_annotations, channels_val, channels_test, hold_out_test_channels):
    dict_val_data = {}
    for clip in list(dict_all_annotations.keys()):
        if clip[:-4].split("_")[0] in hold_out_test_channels:
            continue
        if clip[:-4].split("_")[0] in channels_val:
            dict_val_data[clip] = dict_all_annotations[clip]

    dict_test_data = {}
    for clip in list(dict_all_annotations.keys()):
        if clip[:-4].split("_")[0] in hold_out_test_channels:
            continue
        if clip[:-4].split("_")[0] in channels_test:
            dict_test_data[clip] = dict_all_annotations[clip]

    dict_train_data = {}
    for clip in list(dict_all_annotations.keys()):
        if clip[:-4].split("_")[0] in hold_out_test_channels:
            continue
        if clip[:-4].split("_")[0] not in channels_val + channels_test:
            dict_train_data[clip] = dict_all_annotations[clip]
    return dict_train_data, dict_val_data, dict_test_data


# def create_data_for_model(dict_miniclip_clip_feature, dict_action_embeddings, dict_all_annotations, channel_test,
#                           channel_val, hold_out_test_channels):

def create_data_for_model(dict_miniclip_clip_feature, dict_action_embeddings, dict_train_annotations,
                          dict_val_annotations, dict_test_annotations, balance):
    data_clips_train, data_actions_train, labels_train = [], [], []
    data_clips_val, data_actions_val, labels_val = [], [], []
    data_clips_test, data_actions_test, labels_test = [], [], []

    set_action_miniclip_train = set()
    set_action_miniclip_test = set()
    set_action_miniclip_val = set()

    if balance:
        print("Balance data")

        # with open("data/train_test_val/dict_balanced_annotations_train.json") as f:
        #     dict_train_annotations = json.loads(f.read())
        dict_train_annotations = balance_data(dict_train_annotations)
        with open("data/train_test_val/dict_balanced_annotations_train.json", 'w+') as fp:
            json.dump(dict_train_annotations, fp)

        with open("data/train_test_val/dict_balanced_annotations_val.json") as f:
            dict_val_annotations = json.loads(f.read())
        # dict_val_annotations = balance_data(dict_val_annotations)
        # with open("data/train_test_val/dict_balanced_annotations_val.json", 'w+') as fp:
        #     json.dump(dict_val_annotations, fp)

        with open("data/train_test_val/dict_balanced_annotations_test.json") as f:
            dict_test_annotations = json.loads(f.read())
        # dict_test_annotations = balance_data(dict_test_annotations)
        # with open("data/train_test_val/dict_balanced_annotations_test.json", 'w+') as fp:
        #     json.dump(dict_test_annotations, fp)

    for clip in list(dict_train_annotations.keys()):
        list_action_label = dict_train_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            # action, _ = compute_action(action, use_nouns=False, use_particle=True)
            action_emb = dict_action_embeddings[action]
            # action_emb = np.zeros(1024)
            data_clips_train.append([clip, viz_feat])
            data_actions_train.append([action, action_emb])
            labels_train.append(label)
            set_action_miniclip_train.add(clip[:-8] + ", " + action)

    for clip in list(dict_val_annotations.keys()):
        list_action_label = dict_val_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = dict_action_embeddings[action]
            # action_emb = np.zeros(1024)
            data_clips_val.append([clip, viz_feat])
            data_actions_val.append([action, action_emb])
            labels_val.append(label)
            set_action_miniclip_val.add(clip[:-8] + ", " + action)

    for clip in list(dict_test_annotations.keys()):
        list_action_label = dict_test_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = dict_action_embeddings[action]
            # action_emb = np.zeros(1024)
            data_clips_test.append([clip, viz_feat])
            data_actions_test.append([action, action_emb])
            labels_test.append(label)
            set_action_miniclip_test.add(clip[:-8] + ", " + action)

    print(tabulate([['Total', 'Train', 'Val', 'Test'],
                    [len(data_actions_train) + len(data_actions_val) + len(data_actions_test), len(data_actions_train),
                     len(data_actions_val), len(data_actions_test)]],
                   headers="firstrow"))

    print(Counter(labels_train))
    print(Counter(labels_val))
    print(Counter(labels_test))

    return [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
           [data_clips_test, data_actions_test, labels_test]


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


def baseline_2(train_data, val_data, test_data, model_name, nb_epochs, balance, config_name):
    print("---------- Running " + config_name + " -------------------")

    [data_clips_train, data_actions_train, labels_train, data_actions_names_train], [data_clips_val, data_actions_val,
                                                                                     labels_val,
                                                                                     data_actions_names_val], \
    [data_clips_test, data_actions_test, labels_test, data_actions_names_test] = get_features_from_data(train_data,
                                                                                                        val_data,
                                                                                                        test_data)

    # input_dim_text = data_actions_val[0].shape[0]
    input_dim_text = len(data_actions_train[0])
    input_dim_video = data_clips_train[0].shape[0]
    # input_dim_video = data_clips_train[0].shape

    if config_name.split(" + ")[2] == "finetuned ELMo":
        print("before data_clips_train len: {0}".format(input_dim_video))
        data_actions_train = np.array(data_actions_names_train, dtype=object)[:, np.newaxis]
        data_actions_val = np.array(data_actions_names_val, dtype=object)[:, np.newaxis]
        data_actions_test = np.array(data_actions_names_test, dtype=object)[:, np.newaxis]

        data_clips_train = np.array(data_clips_train, dtype=object)
        data_clips_val = np.array(data_clips_val, dtype=object)
        data_clips_test = np.array(data_clips_test, dtype=object)
        print("after data_clips_train.shape: {0}".format(data_clips_train.shape))

        print("Elmo actions, data_actions_train.shape: {0}".format(data_actions_train.shape))
        finetune_elmo = True
    else:
        finetune_elmo = False

    if config_name.split(" + ")[2] == "finetuned Bert":
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
                                   verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=50)

    now = time.strftime("%c")
    tensorboard = TensorBoard(log_dir="logs/fit/" + now + "_" + config_name, histogram_freq=0, write_graph=True)

    callback_list = [checkpointer]

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
    predicted = list_predictions >= 0
    # predicted = list_predictions > 0.5
    print("before median:")
    print("Predicted test data: " + str(Counter(x for xs in predicted for x in set(xs))))

    # median filter
    predicted = predicted.flatten()
    predicted = scipy.signal.medfilt(predicted,1)
    predicted = predicted.reshape(len(predicted),1)

    print("after median:")
    print("Predicted test data: " + str(Counter(x for xs in predicted for x in set(xs))))

    # list_predictions = model.predict([data_actions_val, data_clips_val])
    # predicted = list_predictions > 0.5

    # acc_test = None
    f1_test = f1_score(labels_test, predicted)
    prec_test = precision_score(labels_test, predicted)
    rec_test = recall_score(labels_test, predicted)
    print("precision {0}, recall: {1}, f1: {2}".format(prec_test, rec_test, f1_test))
    return model_name, acc_train, acc_val, acc_test, f1_test, predicted, list_predictions

def merge_intervals(intervals, overlapping_sec):
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0] - overlapping_sec:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
            merged[-1][2] = max(merged[-1][2], interval[2])

    return merged

def compute_final_proposal(list_all_times):
    groups = group(list_all_times, 3)
    overlapping_sec = 5
    print(groups)
    merged_intervals = merge_intervals(groups, overlapping_sec)

    merged_intervals.sort(key=lambda x: x[2], reverse=True)  # highest scored proposal
    print(merged_intervals)
    proposal = [merged_intervals[0][:-1]]
    return proposal


# def compute_final_proposal(list_all_times):
#     groups = group(list_all_times, 3)
#     print(groups)
#     list_proposals = []
#     (t0_s, t0_e, score0) = groups[0]
#     if len(groups) == 1:
#         list_proposals = [[t0_s, t0_e, score0]]
#     for [t1_s, t1_e, score1] in groups[1:]:
#         if t1_s == t0_e:
#             t0_e = t1_e
#             if score1 > score0:
#                 score0 = score1
#             if len(groups) <= 2:
#                 list_proposals.append([t0_s, t0_e, score0])
#         else:
#             list_proposals.append([t0_s, t0_e, score0])
#             if len(groups) <= 2:
#                 list_proposals.append([t1_s, t1_e, score1])
#             t0_e = t1_e
#             t0_s = t1_s
#             score0 = score1
#         del groups[0]
#
#     list_proposals.sort(key=lambda x: x[2], reverse=True)  # highest scored proposal
#     print(list_proposals)
#     # print(list_proposals)
#     proposal = [list_proposals[0][:-1]]
#     # print("groups1: {0}, proposals: {1}, final: {2}".format(groups1, list_proposals, proposal))
#     return proposal


def compute_predicted_IOU(model_name, predicted_labels_test, test_data, GT, dict_clip_time_per_miniclip,
                          list_predictions=None):
    [data_clips_test, data_actions_test, gt_labels_test] = test_data
    data_clips_test_names = [i[0] for i in data_clips_test]
    data_actions_test_names = [i[0] for i in data_actions_test]
    dict_predicted = {}
    if GT:
        data = zip(data_clips_test_names, data_actions_test_names, gt_labels_test, list_predictions)
        output = "data/results/dict_predicted_GT.json"
    else:
        data = zip(data_clips_test_names, data_actions_test_names, predicted_labels_test, list_predictions)
        output = "data/results/dict_predicted_" + model_name + ".json"

    for [clip, action, label, score] in data:
        miniclip = clip[:-8] + ".mp4"
        if label:
            if miniclip + ", " + action not in dict_predicted.keys():
                dict_predicted[miniclip + ", " + action] = []

            [time_s, time_e] = dict_clip_time_per_miniclip[clip]
            dict_predicted[miniclip + ", " + action].append(time_s)
            dict_predicted[miniclip + ", " + action].append(time_e)
            if not GT:
                dict_predicted[miniclip + ", " + action].append(score[0])
        else:
            if miniclip + ", " + action not in dict_predicted.keys():
                dict_predicted[miniclip + ", " + action] = []

    for key in dict_predicted.keys():
        if not dict_predicted[key]:
            dict_predicted[key].append(-1)
            dict_predicted[key].append(-1)
            if not GT:
                dict_predicted[key].append(-1)

    if GT:
        for key in dict_predicted.keys():
            list_all_times = dict_predicted[key]
            dict_predicted[key] = [[min(list_all_times), max(list_all_times)]]
    else:
        with open("data/results/dict_predicted_GT.json") as f:
            dict_GT = json.loads(f.read())

        for key in list(dict_predicted.keys()):
            list_all_times = dict_predicted[key]
            print(key)
            proposal = compute_final_proposal(list_all_times)
            dict_predicted[key] = proposal

    with open(output, 'w+') as fp:
        json.dump(dict_predicted, fp)


def test_alignment(path_pos_data):
    with open("data/mapped_actions_time_label.json") as f:
        actions_time_label = json.loads(f.read())

    miniclips_list_stemmed_visibile_actions = {}

    # for channel in ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1", "6p0", "6p1", "7p0", "7p1", "8p0", "8p1", "9p0", "9p1", "10p0", "10p1"]:
    for channel in ["1p0", "1p1", "5p0", "5p1"]:
    # for channel in ["1p0"]:

        # extract the visible ones and stem them
        list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions, list_time_visibile, list_time_not_visibile = separate_mapped_visibile_actions(
            actions_time_label, channel)
        list_stemmed_visibile_actions = stemm_list_actions(visible_actions, path_pos_data)

        for index in range(len(list_stemmed_visibile_actions)):
            miniclip = list_miniclips_visibile[index]
            action = list_stemmed_visibile_actions[index]
            time = list_time_visibile[index]
            miniclip_action = miniclip + ", " + action
            miniclips_list_stemmed_visibile_actions[miniclip_action] = [time]

    with open("data/results/dict_predicted_alignment.json", 'w+') as fp:
        json.dump(miniclips_list_stemmed_visibile_actions, fp)


def get_nb_visible_not_visible(dict_val_data):
    nb_visible_actions = 0
    nb_not_visible_actions = 0
    for clip in list(dict_val_data.keys()):
        list_action_label = dict_val_data[clip]
        for [_, label] in list_action_label:
            if label:
                nb_visible_actions += 1
            else:
                nb_not_visible_actions += 1
    return nb_visible_actions, nb_not_visible_actions

def get_list_actions_for_label(dict_video_actions, miniclip, label_type):
    list_type_actions = []
    list_action_labels = dict_video_actions[miniclip]
    for [action, label] in list_action_labels:
        if label == label_type:
            list_type_actions.append(action)
    return list_type_actions


def balance_data(dict_val_data):
    dict_balance_annotation = {}
    nb_visible_actions, nb_not_visible_actions = get_nb_visible_not_visible(dict_val_data)

    if nb_not_visible_actions >= nb_visible_actions:
        ratio_visible_not_visible = int(nb_not_visible_actions / nb_visible_actions)
    else:
        ratio_visible_not_visible = int(nb_visible_actions / nb_not_visible_actions)

    # Downsample data --> delete the non-visible actions
    for video_name in dict_val_data.keys():
        list_not_visible_actions = get_list_actions_for_label(dict_val_data, video_name, False)
        index = 0
        list_all_actions = dict_val_data[video_name]
        for elem in list_not_visible_actions:
            if ratio_visible_not_visible > 1 and index % ratio_visible_not_visible == 0:
                list_all_actions.remove([elem, False])
            index += 1
        dict_balance_annotation[video_name] = list_all_actions

    nb_visible_actions, nb_not_visible_actions = get_nb_visible_not_visible(dict_balance_annotation)
    diff_nb_actions = abs(nb_not_visible_actions - nb_visible_actions)

    while diff_nb_actions:
        # this makes the # actions to vary in Train, Test Eval after each run
        # run it once and save the list
        random_video_name = random.choice(list(dict_balance_annotation))
        list_not_visible_actions = get_list_actions_for_label(dict_balance_annotation, random_video_name, False)
        if list_not_visible_actions:
            list_all_actions = dict_balance_annotation[random_video_name]
            list_all_actions.remove([list_not_visible_actions[0], False])
            diff_nb_actions -= 1

    return dict_balance_annotation

#
# def balance_data(dict_val_data):
#     nb_true = 0
#     nb_false = 0
#     for clip in list(dict_val_data.keys()):
#         list_action_label = dict_val_data[clip]
#         for [_, label] in list_action_label:
#             if label:
#                 nb_true += 1
#             else:
#                 nb_false += 1
#     # print(nb_true)
#     # print(nb_false)
#     ok_count = np.math.ceil(nb_false / nb_true)
#     # print(ok_count)
#     # ok_count = 0
#     # if clip_length == "3s":
#     #     ok_count = 3
#     #     # ok_count = 2
#     # elif clip_length == "10s":
#     #     ok_count = 1
#     dict_balanced_annotations = {}
#     ok_count = ok_count - 1
#     for clip in list(dict_val_data.keys()):
#         list_action_label = dict_val_data[clip]
#         dict_balanced_annotations[clip] = []
#         ok = 0
#         for [action, label] in list_action_label:
#             if label or ok == ok_count:
#                 dict_balanced_annotations[clip].append([action, label])
#             else:
#                 ok += 1
#
#     # nb_true = 0
#     # nb_false = 0
#     # for clip in list(dict_balanced_annotations.keys()):
#     #     list_action_label = dict_balanced_annotations[clip]
#     #     for [_, label] in list_action_label:
#     #         if label:
#     #             nb_true += 1
#     #         else:
#     #             nb_false += 1
#     # print(nb_true)
#     # print(nb_false)
#
#     # with open("data/dict_balanced_annotations.json", 'w+') as fp:
#     #     json.dump(dict_balanced_annotations, fp)
#
#     return dict_balanced_annotations




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--type_action_emb', type=str, choices=["GloVe", "ELMo", "Bert", "DNT"], default="ELMo")
    parser.add_argument('-m', '--model_name', type=str, choices=["MPU", "cosine sim"], default="MPU")
    parser.add_argument('-f', '--finetune', action='store_true')
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('-cl', '--clip_length', type=str, choices=["3s", "10s"], default="3s")
    parser.add_argument('-b', '--balance', type=bool, choices=[True, False], default=True)
    parser.add_argument('-c', '--add_cluster', action='store_true')
    args = parser.parse_args()
    return args


def main():
    set_random_seed()
    args = parse_args()
    hold_out_test_channels = ["9p0", "9p1", "10p0", "10p1"]
    # hold_out_test_channels = []

    # path_pos_data = "/local/oignat/Action_Recog/vlog_action_recognition/data/dict_action_pos_concreteness.json"
    path_pos_data = "data/dict_action_pos_concreteness.json"
    # path_I3D_features = "../i3d_keras/data/results_features_overlapping_" + args.clip_length + "/"
    #
    # path_all_annotations = "data/dict_all_annotations" + args.clip_length + ".json"
    # with open("data/dict_clip_time_per_miniclip" + args.clip_length + ".json") as f:
    #     dict_clip_time_per_miniclip = json.loads(f.read())

    test_alignment(path_pos_data)
    config_name = "alignment"

    channels_test = ["1p0", "1p1", "5p0", "5p1"]
    # channels_test = ["9p0", "9p1", "10p0", "10p1"]
    # channels_val = ["2p0", "2p1", "3p0", "3p1"]
    #
    # # channels_test = ["4p0", "4p1", "6p0", "6p1"]
    # # channels_val = ["1p0", "1p1", "7p0", "7p1"]
    #
    # # channels_test = ["1p0", "1p1"]
    # # channels_val = ["2p0", "2p1", "3p0","3p1"]
    #
    # # channels_test = ["1p0", "1p1"]
    # # channels_val = ["2p0", "2p1"]
    #
    # # load video & text features - time consuming
    # with open(path_all_annotations) as f:
    #     dict_all_annotations = json.load(f)
    #
    # # dict_miniclip_clip_feature = load_data_from_I3D() #if LSTM
    # dict_miniclip_clip_feature = average_i3d_features(path_I3D_features)
    # # dict_action_embeddings = load_text_embeddings(args.type_action_emb, dict_all_annotations)
    #
    # dict_action_embeddings = load_text_embeddings(args.type_action_emb, dict_all_annotations, all_actions=True,
    #                                               use_nouns=False, use_particle=True)
    #
    # if args.add_cluster:
    #     dict_action_embeddings = add_cluster_data(dict_action_embeddings)
    # '''
    #         Create data
    # '''
    # dict_train_annotations, dict_val_annotations, dict_test_annotations = split_data_train_val_test(dict_all_annotations,
    #                                                                                                channels_val,
    #                                                                                                channels_test,
    #                                                                                                hold_out_test_channels)
    #
    # train_data, val_data, test_data = \
    #     create_data_for_model(dict_miniclip_clip_feature, dict_action_embeddings,
    #                           dict_train_annotations, dict_val_annotations, dict_test_annotations, args.balance)
    #
    # '''
    #         Create model
    # '''
    # if args.finetune:
    #     config_name = args.clip_length + " + " + args.model_name + " + " + "finetuned " + args.type_action_emb + " + " + str(
    #         args.epochs)
    #     print("FINETUNING! " + config_name)
    #
    # else:
    #     config_name = args.clip_length + " + " + args.model_name + " + " + args.type_action_emb + " + " + str(
    #         args.epochs)
    # if args.add_cluster:
    #     print("Add cluster info")
    #     config_name = config_name + " + cluster"
    #
    # model_name, acc_train, acc_val, acc_test, f1, predicted, list_predictions = baseline_2(train_data, val_data,
    #                                                                                        test_data,
    #                                                                                        args.model_name,
    #                                                                                        args.epochs, args.balance,
    #                                                                                        config_name)
    # '''
    #     Majority (actions are visible in all clips)
    # '''
    # [_, _, labels_train, _], [_, _, labels_val, _], [_, _, labels_test, _] = get_features_from_data(train_data,
    #                                                                                                 val_data,
    #                                                                                                 test_data)
    # maj_val, maj_labels = compute_majority_label_baseline_acc(labels_train, labels_val)
    # maj_test, maj_labels = compute_majority_label_baseline_acc(labels_train, labels_test)
    # print("maj_val: {:0.2f}".format(maj_val))
    # print("maj_test: {:0.2f}".format(maj_test))
    # print("acc_train: {:0.2f}".format(acc_train))
    # print("acc_val: {:0.2f}".format(acc_val))
    # print("acc_test: {:0.2f}".format(acc_test))
    # print(color.RED + color.BOLD + "f1_test " + color.END + "{:0.2f}".format(f1))
    #
    # '''
    #         Evaluate
    # '''
    GT = False
    if GT:
        predicted = []

    # compute_predicted_IOU(config_name, predicted, test_data, GT, dict_clip_time_per_miniclip,
    #                       list_predictions)
    for channel_test in channels_test:
        if GT:
            method = "GT"
            evaluate(method, channel_test)
        else:
            evaluate(config_name, channel_test)

    ## -------------------------------------------------------------
    # config_name = "majority"
    # compute_predicted_IOU(config_name, maj_labels, val_data, channel_val, GT)
    # if GT:
    #     method = "GT_"
    #     # evaluate(method, channel_test)
    #     evaluate(method, channel_val)
    # else:
    #     # evaluate(args.model_name + "_" + args.type_action_emb, channel_test)
    #     # evaluate(config_name, channel_test)
    #     evaluate(config_name, channel_val)

    # for channel_test in ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1","6p0", "6p1", "7p0", "7p1"]:
    #     # evaluate(args.model_name + "_" + args.type_action_emb, channel_test)
    #     # evaluate("alignment", channel_test)
    #     evaluate("GT_", channel_test)

    # print_results(model_name, acc_train, acc_val, acc_test, maj_val, maj_test)


if __name__ == "__main__":
    main()
