import itertools

from keras import Model

from utils_data_text import create_average_action_embedding, process_data_channel, \
    process_data, create_action_embedding

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Multiply, Add, concatenate, Dropout

from utils_data_video import load_data_from_I3D
import numpy as np

# model similar to TALL (alignment score & regression is different + pre-trained model features used)
def create_MLP_MPU_model(input_dim_video, input_dim_text):
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


def baseline_2(val_actions, val_labels, val_video):
    # get action embedding (both visible & not visible for now)
    list_visibile_actions = []
    for index in range(len(val_actions)):
        action = val_actions[index]
        label = val_labels[index]
        # TODO: Right now get only the GT labeled visible actions
        if label == 0:
            list_visibile_actions.append(action)

    features_matrix_I3D = load_data_from_I3D(miniclip='1p0_1mini_1')
    visible_actions_per_miniclip = create_average_action_embedding(list_visibile_actions)


    # TODO: create data for the model
    list_clips = list(features_matrix_I3D)
    list_actions = list(visible_actions_per_miniclip)

    all_combinations = list(itertools.product(list_actions, list_clips))  # cartesian product
    labels_fiction = [1, 1] + [0] * 10

    list_actions_combined = [a for (a, v) in all_combinations]
    list_clips_combined = [v for (a, v) in all_combinations]

    data_actions_train = np.array(list_actions_combined[:-2])
    data_clips_train = np.array(list_clips_combined[:-2])
    data_actions_train = data_actions_train.reshape(len(all_combinations)-2, 50)
    data_clips_train = data_clips_train.reshape(len(all_combinations)-2, 1024)
    labels_train = labels_fiction[:-2]

    data_actions_test = np.array(list_actions_combined[-2:])
    data_clips_test = np.array(list_clips_combined[-2:])
    data_actions_test = data_actions_test.reshape(2, 50)
    data_clips_test = data_clips_test.reshape(2, 1024)
    labels_test = labels_fiction[-2:]

    input_dim_text = 50
    input_dim_video = 1024

    model = create_MLP_MPU_model(input_dim_video, input_dim_text)
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

    baseline_2(val_actions, val_labels, val_video)


if __name__ == "__main__":
    main()
