import itertools
import json
from collections import Counter

import np
from keras import Model
from tabulate import tabulate

from utils_data_text import create_action_emb, color, print_results

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Activation, Input, Multiply, Add, concatenate, Dropout, Reshape, dot

from utils_data_video import load_data_from_I3D

import tensorflow as tf



# model similar to TALL (alignment score & regression is different + pre-trained model features used)
def create_MPU_model(input_dim_video, input_dim_text):
    action_input = Input(shape=(1,), dtype=tf.string)
    action_output = Dense(64)(action_input)

    video_input = Input(shape=(input_dim_video,), dtype='float32', name='video_input')
    video_output = Dense(64)(video_input)

    # MPU
    multiply = Multiply()([action_output, video_output])
    add = Add()([action_output, video_output])
    concat = concatenate([multiply, add])
    concat = concatenate([concat, action_output])
    concat = concatenate([concat, video_output])

    output = Dense(64, activation='relu')(concat)
    output = Dropout(0.5)(output)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(output)

    model = Model(inputs=[video_input, action_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def create_cosine_sim_model(input_dim_video, input_dim_text):
    action_input = Input(shape=(input_dim_text,), name='action_input')
    action_output = Dense(64)(action_input)

    video_input = Input(shape=(input_dim_video,), dtype='float32', name='video_input')
    video_output = Dense(64)(video_input)

    # now perform the dot product operation to get a similarity measure
    dot_product = dot([action_output, video_output], axes=1, normalize=True)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(dot_product)

    model = Model(inputs=[video_input, action_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model



def create_data_for_model(type_action_emb, path_all_annotations, channel_val, channel_test):
    with open(path_all_annotations) as f:
        dict_all_annotations = json.loads(f.read())

    data_clips_train, data_actions_train, labels_train = [], [], []
    data_clips_val, data_actions_val, labels_val = [], [], []
    data_clips_test, data_actions_test, labels_test = [], [], []

    dict_miniclip_clip_feature = load_data_from_I3D()

    for clip in list(dict_all_annotations.keys()):
        list_action_label = dict_all_annotations[clip]
        # TODO: Spliting the clips, these were extra
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = create_action_emb(action, type_action_emb)
            # if channel_val == clip[:-4].split("_")[0]:
            if "_".join(clip[:-4].split("_")[0:2]) == channel_val:
                data_clips_val.append(viz_feat)
                data_actions_val.append(action_emb)
                labels_val.append(label)
            # elif channel_test == clip[:-4].split("_")[0]:
            elif "_".join(clip[:-4].split("_")[0:2]) == channel_test:
                data_clips_test.append(viz_feat)
                data_actions_test.append(action_emb)
                labels_test.append(label)
            else:
                data_clips_train.append(viz_feat)
                data_actions_train.append(action_emb)
                labels_train.append(label)

    print(tabulate([['Train', 'Val', 'Test'], [len(labels_train), len(labels_val), len(labels_test)]],
                   headers="firstrow"))
    return [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
           [data_clips_test, data_actions_test, labels_test]


def compute_majority_label_baseline_acc(labels_train, labels_test):
    if Counter(labels_train)[False] > Counter(labels_train)[True]:
        maj_label = False
    else:
        maj_label = True
    maj_labels = [maj_label] * len(labels_test)
    nb_correct = 0
    for i in range(len(labels_test)):
        if maj_labels[i] == labels_test[i]:
            nb_correct += 1
    return nb_correct / len(labels_test)


def baseline_2(train_data, val_data, test_data, model_name, type_action_emb):
    print("---------- Running " + model_name + " + " + type_action_emb + " -------------------")

    [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
    [data_clips_test, data_actions_test, labels_test] = train_data, val_data, test_data

    input_dim_text = 0
    #TODO: Automatize
    if type_action_emb == "GloVe":
        input_dim_text = 50
    if type_action_emb == "ELMo":
        input_dim_text = 1024
    input_dim_video = 1024

    if model_name == "MPU":
        model = create_MPU_model(input_dim_video, input_dim_text)
    elif model_name == "cosine sim":
        model = create_cosine_sim_model(input_dim_video, input_dim_text)
    else:
        raise ValueError("Wrong model name!")

    checkpointer = ModelCheckpoint(monitor='val_acc',
                                   filepath='data/Model_params/' + model_name + " + " + type_action_emb + '.hdf5',
                                   verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=10)
    callback_list = [checkpointer, earlystopper]

    model.fit([data_clips_train, data_actions_train], labels_train,
              validation_data=([data_clips_val, data_actions_val], labels_val),
              epochs=20, batch_size=32, callbacks=callback_list)

    score, acc_train = model.evaluate([data_clips_train, data_actions_train], labels_train)
    score, acc_val = model.evaluate([data_clips_val, data_actions_val], labels_val)
    score, acc_test = model.evaluate([data_clips_test, data_actions_test], labels_test)

    predicted = model.predict([data_clips_test, data_actions_test]) > 0.5
    maj_val = compute_majority_label_baseline_acc(labels_train, labels_val)
    maj_test = compute_majority_label_baseline_acc(labels_train, labels_test)

    return model_name, acc_train, acc_val, acc_test, maj_val, maj_test



def main():
    type_action_emb_list = ["GloVe", "ELMo"]
    type_action_emb = type_action_emb_list[1]
    model_name_list = ["MPU", "cosine sim"]
    model_name = model_name_list[0]

    train_data, val_data, test_data = create_data_for_model(type_action_emb,
                                                            path_all_annotations="data/dict_all_annotations.json",
                                                            channel_val="1p0_10mini", channel_test="1p0_1mini")

    model_name, acc_train, acc_val, acc_test, maj_val, maj_test = baseline_2(train_data, val_data, test_data, model_name, type_action_emb)
    print_results(model_name, acc_train, acc_val, acc_test, maj_val, maj_test)


if __name__ == "__main__":
    main()
