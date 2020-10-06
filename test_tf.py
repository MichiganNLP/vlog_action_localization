import os
import time
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from utils_data_text import get_features_from_data, read_class_results

print(tf.__version__)


# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.tables_initializer())


# The provided .npy file thus has shape (1, num_frames, 224, 224, 3) for RGB, corresponding to a batch size of 1
# def load_video_feat():
#     path_I3D_features = "../i3d_keras/data/results_overlapping/"
#     #path_I3D_features = "test_rgb.npy"
#     print("loading I3D")
#     list_features = []
#     for filename in tqdm(os.listdir(path_I3D_features)):
#         print(path_I3D_features + filename)
#         features = np.load(path_I3D_features + filename)
#         list_features.append(features)
#         print(features.shape)
#     return list_features

def load_video_feat(clip):
    filename = clip[:-4] + "_rgb.npy"
    path_I3D_features = "../i3d_keras/data/results_overlapping/"
    # print("loading I3D")
    try:
        features = np.load(path_I3D_features + filename)
    except Exception as e:
        print(clip)
        print(e)
        return np.zeros((1, 64, 224, 224, 3))

    # features = np.load("test_rgb.npy")
    return features


# def load_video_feat():
#     path_I3D_features = "../i3d_keras/data/results_overlapping/"
#     print("loading I3D")
#     dict_clip_feat = {}
#     for filename in tqdm(os.listdir(path_I3D_features)):
#         if filename.split("_")[0] not in ["1p0", "1p1", "5p0", "5p1"]:
#             continue
#         try:
#             features = np.load(path_I3D_features + filename)
#         except Exception as e:
#             print(filename)
#             print(e)
#
#         dict_clip_feat[filename[:-8] + ".mp4"] = features
#     return features

def finetune_howto1m(train_data, val_data, test_data):
    # hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2",
    #                            input_shape=[], dtype=tf.string, trainable=True)
    #
    # model = keras.Sequential()
    # model.add(hub_layer)
    # model.add(keras.layers.Dense(16, activation='relu'))
    # model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    # model.summary()
    # text_values_list = ['java', 'hadoop', '.net']
    # labels = [0, 0, 1]
    # model.fit(x=text_values_list, y=labels, epochs=100)

    # [data_clips_train, data_actions_train, labels_train, data_actions_names_train, data_clips_names_train], [
    #     data_clips_val, data_actions_val,
    #     labels_val,
    #     data_actions_names_val, data_clips_names_val], \
    # [data_clips_test, data_actions_test, labels_test, data_actions_names_test,
    #  data_clips_names_test] = get_features_from_data(train_data,
    #                                                  val_data,
    #                                                  test_data)

    hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1",
                               output_shape=[128],
                               input_shape=[], dtype=tf.string)

    model = keras.Sequential()
    model.add(hub_layer)
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    # module = hub.load("https://tfhub.dev/deepmind/mil-nce/i3d/1", tags={"train"})
    # print(module.get_signature_names())
    # print(module.get_input_info_dict())
    # # inputs_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
    # input_frames = tf.placeholder(tf.float32, shape=(None, None, None, None, 3))
    # # inputs_words are just a list of sentences (i.e. ['the sky is blue', 'someone cutting an apple'])
    # input_words = tf.placeholder(tf.string, shape=(None,))
    #
    # vision_output = module(input_frames, signature='video', as_dict=True)
    # text_output = module(input_words, signature='text', as_dict=True)
    #
    # video_embedding = vision_output['video_embedding']
    # text_embedding = text_output['text_embedding']

    # hub_layer = hub.KerasLayer(module_obj)
    # model = keras.Sequential()
    # model.add(hub_layer)
    # model.add(keras.layers.Dense(16, activation='relu'))
    # model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    # model.summary()
    # # A simple classification problem, briging related wordds together.
    # # test = ['hive', 'hadoop', '.net']
    # # labels = [0, 0, 1]
    # # model.fit(x=text_values_list, y=labels, epochs=50)
    # file_path_best_model = 'model/Model_params/not_vis_howto100m_finetuned.hdf5'
    #
    # checkpointer = ModelCheckpoint(monitor='val_acc',
    #                                filepath=file_path_best_model,
    #                                save_best_only=True, save_weights_only=True)
    # earlystopper = EarlyStopping(monitor='val_acc', patience=15)
    # tensorboard = TensorBoard(log_dir="logs/fit/" + time.strftime("%c") + "_howto100m", histogram_freq=0,
    #                           write_graph=True)
    # callback_list = [earlystopper, checkpointer]
    #
    # model.fit([data_actions_train, data_clips_train], labels_train,
    #           validation_data=([data_actions_val, data_clips_val], labels_val),
    #           epochs=65, batch_size=64, verbose=1, callbacks=callback_list)
    # model.summary()
    #
    # os.makedirs("finetuned_module_export", exist_ok=True)
    # export_module_dir = os.path.join(os.getcwd(), "finetuned_module_export")
    # tf.saved_model.save(module_obj, export_module_dir)
    #
    # # print("Load best model weights from " + file_path_best_model)
    # # model.load_weights(file_path_best_model)


def method_tf_actions(train_data, val_data, test_data):
    [data_clips_feat_train, data_actions_emb_train, labels_train, data_actions_names_train,
     data_clips_names_train], [data_clips_feat_val, data_actions_emb_val, labels_val, data_actions_names_val,
                               data_clips_names_val], [
        data_clips_feat_test, data_actions_emb_test, labels_test, data_actions_names_test, data_clips_names_test] = \
        get_features_from_data(train_data, val_data, test_data)

    predicted = []

    # dict_clip_feat = load_video_feat()

    # inputs_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
    input_frames = tf.placeholder(tf.float32, shape=(None, None, None, None, 3))
    # inputs_words are just a list of sentences (i.e. ['the sky is blue', 'someone cutting an apple'])
    input_words = tf.placeholder(tf.string, shape=(None,))

    # module = hub.Module("https://tfhub.dev/deepmind/mil-nce/s3d/1")
    module = hub.Module("https://tfhub.dev/deepmind/mil-nce/i3d/1")
    # module = hub.Module("https://tfhub.dev/deepmind/mil-nce/i3d/1", trainable=True, tags={"train"})


    vision_output = module(input_frames, signature='video', as_dict=True)
    text_output = module(input_words, signature='text', as_dict=True)

    video_embedding = vision_output['video_embedding']
    text_embedding = text_output['text_embedding']
    # We compute all the pairwise similarity scores between video and text.
    similarity_matrix = tf.matmul(text_embedding, video_embedding, transpose_b=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())


    for [action, clip] in tqdm(list(zip(data_actions_names_test, data_clips_names_test))):#[:100]):
        # clip_feat_rgb = dict_clip_feat[clip]
        clip_feat_rgb = load_video_feat(clip)

        result_sim = sess.run([similarity_matrix], feed_dict={input_words: [action],
                                                              input_frames: clip_feat_rgb})

        predicted.append(result_sim)

        # list_actions_per_clip = [action]
        # clip_0 = clip

    # np.save("data/tf_tes_predicted_train.npy", predicted)
    np.save("data/tf_tes_predicted.npy", predicted)
    # np.save("data/tf_train_predicted.npy", predicted)
    # print("Predicted " + str(Counter(predicted)))
    # f1_test = f1_score(labels_test, predicted)
    # prec_test = precision_score(labels_test, predicted)
    # rec_test = recall_score(labels_test, predicted)
    # acc_test = accuracy_score(labels_test, predicted)
    # print("precision {0}, recall: {1}, f1: {2}".format(prec_test, rec_test, f1_test))
    # print("acc_test: {:0.2f}".format(acc_test))
    #
    # list_predictions = predicted
    # return predicted, list_predictions
    return [], []


#
# def run_tf(clip_feat_rgb, list_actions_per_clip):
#
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.tables_initializer())
#         similarity_matrix_res = sess.run([similarity_matrix], feed_dict={input_words: list_actions_per_clip, input_frames: clip_feat_rgb})
#
#     return similarity_matrix_res


def read_test_predicted(train_data, val_data, test_data):
    [data_clips_train, data_actions_train, labels_train, data_actions_names_train, data_clips_names_train], [
        data_clips_val, data_actions_val,
        labels_val,
        data_actions_names_val, data_clips_names_val], \
    [data_clips_test, data_actions_test, labels_test, data_actions_names_test,
     data_clips_names_test] = get_features_from_data(train_data,
                                                     val_data,
                                                     test_data)

    content = np.load("data/tf_tes_predicted.npy")
    predicted = np.squeeze(content)
    normalized_predicted = []
    print(predicted)
    print(predicted)
    # learn the threshold
    normalized_threshold = (min(predicted) + max(predicted))  /2
    print(min(predicted))
    print(max(predicted))
    print(normalized_threshold)

    for i in predicted:
        if i >= normalized_threshold:
            normalized_predicted.append(True)
        else:
            normalized_predicted.append(False)
    predicted = normalized_predicted
    # for action, clip, label_gt, label_pred in tqdm(
    #         list(zip(data_actions_names_test, data_clips_names_test, labels_test, predicted))[50:150]):
    #     print(action + " ; " + clip + " ; " + str(label_gt) + " ; " + str(label_pred))

    print("Predicted " + str(Counter(predicted)))
    f1_test = f1_score(labels_test, predicted)
    prec_test = precision_score(labels_test, predicted)
    rec_test = recall_score(labels_test, predicted)
    acc_test = accuracy_score(labels_test, predicted)
    print("precision {0}, recall: {1}, f1: {2}".format(prec_test, rec_test, f1_test))
    print("acc_test: {:0.2f}".format(acc_test))

    list_predictions = predicted
    return predicted, list_predictions


def main():
    read_test_predicted()
    # video_rgb = load_video_feat()

    # # inputs_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
    # input_frames = tf.placeholder(tf.float32, shape=(None, None, None, None, 3))
    # # inputs_words are just a list of sentences (i.e. ['the sky is blue', 'someone cutting an apple'])
    # input_words = tf.placeholder(tf.string, shape=(None,))
    #
    # module = hub.Module("https://tfhub.dev/deepmind/mil-nce/s3d/1")
    #
    # vision_output = module(input_frames, signature='video', as_dict=True)
    # text_output = module(input_words, signature='text', as_dict=True)
    #
    # video_embedding = vision_output['video_embedding']
    # text_embedding = text_output['text_embedding']
    # # We compute all the pairwise similarity scores between video and text.
    # similarity_matrix = tf.matmul(text_embedding, video_embedding, transpose_b=True)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     sess.run(tf.tables_initializer())
    #
    #     print(sess.run([similarity_matrix], feed_dict={input_words: ['the sky is blue'], input_frames: video_rgb}))


if __name__ == "__main__":
    main()
