import os

from sklearn.metrics import f1_score

from compute_text_embeddings import create_glove_embeddings, create_bert_embeddings, create_elmo_embddings, \
    ElmoEmbeddingLayer
from evaluation import evaluate
import json
import numpy as np
from collections import Counter
from keras import Model
from tabulate import tabulate
import time
from utils_data_text import group, get_features_from_data, stemm_list_actions, \
    separate_mapped_visibile_actions
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Input, Multiply, Add, concatenate, Dropout, Reshape, dot, LSTM
from utils_data_video import load_data_from_I3D, average_i3d_features
import argparse

import tensorflow as tf
from keras import backend as K, Model
from keras import layers

# # Initialize session
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), allow_soft_placement=True)
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
K.set_session(sess)


def set_random_seed():
    # Set a seed value
    seed_value = 12321
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
def create_MPU_model(input_dim_video, input_dim_text, finetune):
    # Second input
    if finetune:
        action_input = layers.Input(shape=(1,), dtype=tf.string, name='action_input')
        action_emb = ElmoEmbeddingLayer()(action_input)
        action_output = Dense(64)(action_emb)

    else:
        print("no finetune")
        action_input = Input(shape=(input_dim_text,), name='action_input')
        action_output = Dense(64)(action_input)

    video_input = Input(shape=(input_dim_video,), dtype='float32', name='video_input')
    # video_input = Input(shape=(input_dim_video[0], input_dim_video[1],), dtype='float32', name='video_input')
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    # video_output = LSTM(64)(video_input)

    video_output = Dense(64)(video_input)

    # MPU
    multiply = Multiply()([action_output, video_output])
    add = Add()([action_output, video_output])
    concat_multiply_add = concatenate([multiply, add])

    concat = concatenate([action_output, video_output])
    FC = Dense(64)(concat)

    concat_all = concatenate([concat_multiply_add, FC])

    output = Dense(64)(concat_all)
    output = Dropout(0.5)(output)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(output)

    model = Model(inputs=[action_input, video_input], outputs=[main_output])
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


def load_text_embeddings(type_action_emb, dict_all_annotations):
    set_actions = set()
    for clip in list(dict_all_annotations.keys()):
        list_action_label = dict_all_annotations[clip]
        for [action, _] in list_action_label:
            set_actions.add(action)
    list_all_actions = list(set_actions)

    if type_action_emb == "GloVe":
        return create_glove_embeddings(list_all_actions)
    elif type_action_emb == "ELMo":
        with open('data/dict_action_embeddings_ELMo.json') as f:
            json_load = json.loads(f.read())
        return json_load
    # return create_elmo_embddings(list_all_actions)  # if need to create new
    elif type_action_emb == "Bert":
        # with open('data/dict_action_embeddings_Bert.json') as f:
        #     json_load = json.loads(f.read())
        # return json_load
        return create_bert_embeddings(list_all_actions)
    elif type_action_emb == "DNT":
        with open('steve_human_action/dict_action_emb_DNT.json') as f:
            json_load = json.loads(f.read())
        return json_load
    else:
        raise ValueError("Wrong action emb type")


def create_data_for_model(dict_miniclip_clip_feature, dict_action_embeddings, dict_all_annotations, channel_test,
                          channel_val):
    data_clips_train, data_actions_train, labels_train = [], [], []
    data_clips_val, data_actions_val, labels_val = [], [], []
    data_clips_test, data_actions_test, labels_test = [], [], []

    set_action_miniclip_train = set()
    set_action_miniclip_test = set()
    set_action_miniclip_val = set()

    for clip in list(dict_all_annotations.keys()):
        list_action_label = dict_all_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = dict_action_embeddings[action]
            # action_emb = np.zeros(1024)
            # for debug
            # if clip[:-4].split("_")[0] not in {"1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1",
            #                                    "6p0", "6p1", "7p0", "7p1"}:
            #     continue

            if clip[:-4].split("_")[0] in channel_val:
                # if "_".join(clip[:-4].split("_")[0:2]) == channel_val:
                data_clips_val.append([clip, viz_feat])
                data_actions_val.append([action, action_emb])
                labels_val.append(label)
                set_action_miniclip_val.add(clip[:-8] + ", " + action)

            elif clip[:-4].split("_")[0] in channel_test:
                # elif "_".join(clip[:-4].split("_")[0:2]) == channel_test:
                data_clips_test.append([clip, viz_feat])
                data_actions_test.append([action, action_emb])
                labels_test.append(label)
                set_action_miniclip_test.add(clip[:-8] + ", " + action)
            else:
                data_clips_train.append([clip, viz_feat])
                data_actions_train.append([action, action_emb])
                labels_train.append(label)
                set_action_miniclip_train.add(clip[:-8] + ", " + action)
        ## for debug
        # if len(labels_val) > 0 and len(labels_test) > 0 and len(labels_train) > 0:
        #     break

    print(tabulate([['Train', 'Val', 'Test'],
                    # [len(set_action_miniclip_train), len(set_action_miniclip_val), len(set_action_miniclip_test)]],
                    [len(data_actions_train), len(data_actions_val), len(data_actions_test)]],
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

    if config_name.split(" + ")[1] == "finetuned ELMo":
        print("before data_clips_train len: {0}".format(input_dim_video))
        data_actions_train = np.array(data_actions_names_train, dtype=object)[:, np.newaxis]
        data_actions_val = np.array(data_actions_names_val, dtype=object)[:, np.newaxis]
        data_actions_test = np.array(data_actions_names_test, dtype=object)[:, np.newaxis]

        data_clips_train = np.array(data_clips_train, dtype=object)
        data_clips_val = np.array(data_clips_val, dtype=object)
        data_clips_test = np.array(data_clips_test, dtype=object)
        print("after data_clips_train.shape: {0}".format(data_clips_train.shape))

        print("Elmo actions, data_actions_train.shape: {0}".format(data_actions_train.shape))
        finetune = True
    else:
        finetune = False

    if model_name == "MPU":
        model = create_MPU_model(input_dim_video, input_dim_text, finetune)
    elif model_name == "cosine sim":
        model = create_cosine_sim_model(input_dim_video, input_dim_text)
    else:
        raise ValueError("Wrong model name!")

    if balance:
        file_path_best_model = 'data/Model_params/' + config_name + '.hdf5'
    else:
        file_path_best_model = 'data/model_params_unbalanced/' + config_name + '.hdf5'
    checkpointer = ModelCheckpoint(monitor='val_acc',
                                   filepath=file_path_best_model,
                                   verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=50)

    now = time.strftime("%c")
    tensorboard = TensorBoard(log_dir="logs/fit/" + now + "_" + config_name, histogram_freq=0, write_graph=True)

    callback_list = [checkpointer, tensorboard]

    if not os.path.isfile(file_path_best_model):
        model.fit([data_actions_train, data_clips_train], labels_train,
                  validation_data=([data_actions_val, data_clips_val], labels_val),
                  epochs=nb_epochs, batch_size=64, callbacks=callback_list)

    print("Load best model weights from " + file_path_best_model)
    model.load_weights(file_path_best_model)

    score, acc_train = model.evaluate([data_actions_train, data_clips_train], labels_train)
    score, acc_test = model.evaluate([data_actions_test, data_clips_test], labels_test)
    list_predictions = model.predict([data_actions_test, data_clips_test])
    predicted = list_predictions > 0.5
    print(Counter(x for xs in predicted for x in set(xs)))
    print(Counter(labels_test))
    score, acc_val = model.evaluate([data_actions_val, data_clips_val], labels_val)
    # list_predictions = model.predict([data_actions_val, data_clips_val])
    # predicted = list_predictions > 0.5

    # acc_test = None
    f1_test = f1_score(labels_test, predicted)
    return model_name, acc_train, acc_val, acc_test, f1_test, predicted, list_predictions


def compute_final_proposal(list_all_times):
    groups = group(list_all_times, 3)
    groups1 = group(list_all_times, 3)
    list_proposals = []
    (t0_s, t0_e, score0) = groups[0]
    if len(groups) == 1:
        list_proposals = [[t0_s, t0_e, score0]]
    for [t1_s, t1_e, score1] in groups[1:]:
        if t1_s == t0_e:
            t0_e = t1_e
            if score1 > score0:
                score0 = score1
            if len(groups) <= 2:
                list_proposals.append([t0_s, t0_e, score0])
        else:
            list_proposals.append([t0_s, t0_e, score0])
            if len(groups) <= 2:
                list_proposals.append([t1_s, t1_e, score1])
            t0_e = t1_e
            t0_s = t1_s
            score0 = score1
        del groups[0]

    list_proposals.sort(key=lambda x: x[2], reverse=True)  # highest scored proposal
    proposal = [list_proposals[0][:-1]]
    # print("groups1: {0}, proposals: {1}, final: {2}".format(groups1, list_proposals, proposal))
    return proposal


def compute_predicted_IOU(model_name, predicted_labels_test, test_data, channel, GT, dict_clip_time_per_miniclip,
                          list_predictions=None):
    [data_clips_test, data_actions_test, gt_labels_test] = test_data
    data_clips_test_names = [i[0] for i in data_clips_test]
    data_actions_test_names = [i[0] for i in data_actions_test]

    dict_predicted_GT = {}
    if GT:
        data = zip(data_clips_test_names, data_actions_test_names, gt_labels_test, list_predictions)
        output = "data/results/dict_predicted_GT_" + channel + ".json"
    else:
        data = zip(data_clips_test_names, data_actions_test_names, predicted_labels_test, list_predictions)
        output = "data/results/dict_predicted_" + model_name + channel + ".json"

    for [clip, action, label, score] in data:
        miniclip = clip[:-8] + ".mp4"
        if label:
            if miniclip + ", " + action not in dict_predicted_GT.keys():
                dict_predicted_GT[miniclip + ", " + action] = []

            [time_s, time_e] = dict_clip_time_per_miniclip[clip]
            dict_predicted_GT[miniclip + ", " + action].append(time_s)
            dict_predicted_GT[miniclip + ", " + action].append(time_e)
            dict_predicted_GT[miniclip + ", " + action].append(score[0])
        else:
            if miniclip + ", " + action not in dict_predicted_GT.keys():
                dict_predicted_GT[miniclip + ", " + action] = []

    for key in dict_predicted_GT.keys():
        if not dict_predicted_GT[key]:
            dict_predicted_GT[key].append(-1)
            dict_predicted_GT[key].append(-1)
            dict_predicted_GT[key].append(-1)

    if GT:
        for key in dict_predicted_GT.keys():
            # TODO: REMOVE SCORE
            list_all_times = dict_predicted_GT[key]
            dict_predicted_GT[key] = [[min(list_all_times), max(list_all_times)]]
    else:
        for key in list(dict_predicted_GT.keys()):
            list_all_times = dict_predicted_GT[key]
            # dict_predicted_GT[key] = group(list_all_times, 2)
            dict_predicted_GT[key] = compute_final_proposal(list_all_times)

        # for key in dict_predicted_GT.keys():
        #     list_all_times = dict_predicted_GT[key]
        #     dict_predicted_GT[key] = [[min(list_all_times), max(list_all_times)]]

    with open(output, 'w+') as fp:
        json.dump(dict_predicted_GT, fp)


def test_alignment(path_pos_data):
    with open("data/mapped_actions_time_label.json") as f:
        actions_time_label = json.loads(f.read())

    for channel in ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1", "6p0", "6p1", "7p0", "7p1"]:

        # extract the visible ones and stem them
        list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions, list_time_visibile, list_time_not_visibile = separate_mapped_visibile_actions(
            actions_time_label, channel)
        list_stemmed_visibile_actions = stemm_list_actions(visible_actions, path_pos_data)

        miniclips_list_stemmed_visibile_actions = {}
        for index in range(len(list_stemmed_visibile_actions)):
            miniclip = list_miniclips_visibile[index]
            action = list_stemmed_visibile_actions[index]
            time = list_time_visibile[index]
            miniclip_action = miniclip + ", " + action
            miniclips_list_stemmed_visibile_actions[miniclip_action] = [time]

        with open("data/results/dict_predicted_alignment" + channel + ".json", 'w+') as fp:
            json.dump(miniclips_list_stemmed_visibile_actions, fp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--type_action_emb', type=str, choices=["GloVe", "ELMo", "Bert", "DNT"], default="ELMo")
    parser.add_argument('-m', '--model_name', type=str, choices=["MPU", "cosine sim"], default="MPU")
    parser.add_argument('-f', '--finetune', type=bool, choices=[True, False], default=False)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('-cl', '--clip_length', type=str, choices=["3s", "10s"], default="3s")
    parser.add_argument('-b', '--balance', type=bool, choices=[True, False], default=True)
    args = parser.parse_args()
    return args

def balance_data(dict_all_annotations, clip_length):
    ok_count = 0
    if clip_length == "3s":
        ok_count = 2
    elif clip_length == "10s":
        ok_count = 1
    dict_balanced_annotations = {}
    for clip in list(dict_all_annotations.keys()):
        list_action_label = dict_all_annotations[clip]
        dict_balanced_annotations[clip] = []
        ok = 0
        for [action, label] in list_action_label:
            if label or ok == ok_count:
                dict_balanced_annotations[clip].append([action,label])
            else:
                ok += 1
    return dict_balanced_annotations

def main():
    set_random_seed()
    args = parse_args()

    path_pos_data = "/local/oignat/Action_Recog/vlog_action_recognition/data/dict_action_pos_concreteness.json"
    # path_all_annotations = "data/dict_all_annotations.json"
    path_all_annotations = "data/dict_all_annotations" + args.clip_length + ".json"
    with open("data/dict_clip_time_per_miniclip" + args.clip_length + ".json") as f:
        dict_clip_time_per_miniclip = json.loads(f.read())

    # test_alignment(path_pos_data)

    channels_test = ["1p0"]
    channels_val = ["1p1", "3p0"]

    # load video & text features - time consuming
    with open(path_all_annotations) as f:
        dict_all_annotations = json.load(f)

    if args.balance:
        print("Balance data")
        dict_all_annotations = balance_data(dict_all_annotations, args.clip_length)

    # dict_miniclip_clip_feature = load_data_from_I3D() #if LSTM
    dict_miniclip_clip_feature = average_i3d_features()
    dict_action_embeddings = load_text_embeddings(args.type_action_emb, dict_all_annotations)



    GT = False
    # for channel_test in ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1", "6p0", "6p1", "7p0", "7p1"]:
    '''
            Create data
    '''
    train_data, val_data, test_data = \
        create_data_for_model(dict_miniclip_clip_feature, dict_action_embeddings, dict_all_annotations,
                              channels_test, channels_val)



    '''
            Create model
    '''
    if args.finetune:
        config_name = args.clip_length + " + " + args.model_name + " + " + "finetuned " + args.type_action_emb + " + " + str(
            args.epochs)
    else:
        config_name = args.clip_length + " + " + args.model_name + " + " + args.type_action_emb + " + " + str(
            args.epochs)
    model_name, acc_train, acc_val, acc_test, f1, predicted, list_predictions = baseline_2(train_data, val_data,
                                                                                           test_data,
                                                                                           args.model_name,
                                                                                           args.epochs, args.balance,
                                                                                           config_name)
    '''
        Majority (actions are visible in all clips)
    '''
    [_, _, labels_train, _], [_, _, labels_val, _], [_, _, labels_test, _] = get_features_from_data(train_data,
                                                                                                    val_data,
                                                                                                    test_data)
    maj_val, maj_labels = compute_majority_label_baseline_acc(labels_train, labels_val)
    maj_test, maj_labels = compute_majority_label_baseline_acc(labels_train, labels_test)
    print("maj_val: {:0.2f}".format(maj_val))
    print("maj_test: {:0.2f}".format(maj_test))
    print("acc_val: {:0.2f}".format(acc_val))
    print("acc_test: {:0.2f}".format(acc_test))
    print("f1_test: {:0.2f}".format(f1))

    '''
            Evaluate
    '''
    # predicted = []
    # GT = True
    for channel_test in channels_test:
        compute_predicted_IOU(config_name, predicted, test_data, channel_test, GT, dict_clip_time_per_miniclip,
                              list_predictions)
        if GT:
            method = "GT_"
            # evaluate(method, channel_test)
            evaluate(method, channel_test)
        else:
            # evaluate(args.model_name + "_" + args.type_action_emb, channel_test)
            # evaluate(config_name, channel_test)
            evaluate(config_name, channel_test)
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
