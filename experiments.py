import itertools

from keras import Model

from utils_data_text import create_average_action_embedding, process_data_channel, \
    process_data, create_action_embedding

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Multiply, Add, concatenate, Dropout, Reshape, dot

from utils_data_video import load_data_from_I3D
import numpy as np


# model similar to TALL (alignment score & regression is different + pre-trained model features used)
def create_MPU_model(input_dim_video, input_dim_text):
    action_input = Input(shape=(input_dim_text,), name='action_input')
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


def create_data_for_model(clip_features, action_features):
    list_clip_features = list(clip_features)
    list_action_features = list(action_features)

    all_combinations = list(itertools.product(list_action_features, list_clip_features))  # cartesian product
    labels_fiction = [1, 1] + [0] * 10

    list_actions_combined = [a for (a, v) in all_combinations]
    list_clips_combined = [v for (a, v) in all_combinations]

    data_actions_train = np.array(list_actions_combined[:-2])
    data_clips_train = np.array(list_clips_combined[:-2])
    data_actions_train = data_actions_train.reshape(len(all_combinations) - 2, 50)
    data_clips_train = data_clips_train.reshape(len(all_combinations) - 2, 1024)
    labels_train = labels_fiction[:-2]

    data_actions_test = np.array(list_actions_combined[-2:])
    data_clips_test = np.array(list_clips_combined[-2:])
    data_actions_test = data_actions_test.reshape(2, 50)
    data_clips_test = data_clips_test.reshape(2, 1024)
    labels_test = labels_fiction[-2:]

    return [data_clips_train, data_actions_train, labels_train], [data_clips_test, data_actions_test, labels_test]


def baseline_2(val_actions, val_labels, val_video, model_name):
    # get action embedding (both visible & not visible for now)
    list_visibile_actions = []
    for index in range(len(val_actions)):
        action = val_actions[index]
        label = val_labels[index]
        # TODO: Right now get only the GT labeled visible actions
        if label == 0:
            list_visibile_actions.append(action)

    clip_features_I3D = load_data_from_I3D(miniclip='1p0_1mini_1')
    visible_actions_GloVe = create_average_action_embedding(list_visibile_actions)

    [data_clips_train, data_actions_train, labels_train], [data_clips_test, data_actions_test,
                                                           labels_test] = create_data_for_model(clip_features_I3D,
                                                                                                visible_actions_GloVe)

    input_dim_text = 50
    input_dim_video = 1024

    if model_name == "MPU":
        model = create_MPU_model(input_dim_video, input_dim_text)
    elif model_name == "cosine sim":
        model = create_cosine_sim_model(input_dim_video, input_dim_text)
    else:
        raise ValueError("Wrong model name!")

    model.fit([data_clips_train, data_actions_train], labels_train,
              epochs=2, batch_size=32)

    score, acc_train = model.evaluate([data_clips_train, data_actions_train], labels_train)
    score, acc_test = model.evaluate([data_clips_test, data_actions_test], labels_test)

    predicted = model.predict([data_clips_test, data_actions_test]) > 0.5
    print(acc_train)
    print(acc_test)
    print(predicted)


def main():
    # do this just once!
    dict_video_actions, train_data, test_data, val_data = process_data_channel()

    [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], [train_video,
                                                                                          test_video, val_video] = \
        process_data(train_data, test_data, val_data)

    #baseline_2(val_actions, val_labels, val_video, model_name="MPU")
    baseline_2(val_actions, val_labels, val_video, model_name="cosine sim")


if __name__ == "__main__":
    main()
