import json
import os
import numpy as np
import math
from tabulate import tabulate
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

'''
    Data
'''

def average_i3d_features():
    with open('data/embeddings/dict_I3D_avg.json') as json_file:
        dict_clip_feature = json.load(json_file)
    return dict_clip_feature

def load_text_embeddings(type_action_emb):
    if type_action_emb == "ELMo":
        with open('data/embeddings/dict_action_embeddings_ELMo.json') as f:
            json_load = json.loads(f.read())
        return json_load
    elif type_action_emb == "Bert":
        with open('data/embeddings/dict_action_embeddings_Bert2.json') as f:
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


def create_data_for_model(type_action_emb, balance,
                          path_all_annotations,
                          channels_val,
                          channels_test, hold_out_test_channels):
    with open(path_all_annotations) as f:
        dict_all_annotations = json.load(f)

    dict_miniclip_clip_feature = average_i3d_features()
    dict_action_embeddings = load_text_embeddings(type_action_emb)

    dict_train_annotations, dict_val_annotations, dict_test_annotations = split_data_train_val_test(
        dict_all_annotations,
        channels_val,
        channels_test,
        hold_out_test_channels)

    data_clips_train, data_actions_train, labels_train = [], [], []
    data_clips_val, data_actions_val, labels_val = [], [], []
    data_clips_test, data_actions_test, labels_test = [], [], []

    set_action_miniclip_train, set_action_miniclip_test, set_action_miniclip_val = set(), set(), set()
    set_videos_train, set_videos_test, set_videos_val = set(), set(), set()
    set_miniclip_train, set_miniclip_test, set_miniclip_val = set(), set(), set()
    set_clip_train, set_clip_test, set_clip_val = set(), set(), set()
    clip_action_label_train, clip_action_label_test, clip_action_label_val = {}, {}, {}

    if balance:
        print("Balance data (train & val)")

        with open("data/train_test_val/dict_balanced_annotations_train_9_10.json") as f:
            dict_train_annotations = json.loads(f.read())

        with open("data/train_test_val/dict_balanced_annotations_val_9_10.json") as f:
            dict_val_annotations = json.loads(f.read())

    for clip in list(dict_train_annotations.keys()):
        list_action_label = dict_train_annotations[clip]
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        # viz_feat = np.array(dict_miniclip_clip_feature[clip[:-4]])
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = dict_action_embeddings[action]

            data_clips_train.append([clip, viz_feat])
            data_actions_train.append([action, action_emb])
            labels_train.append(label)
            set_action_miniclip_train.add(clip[:-8] + ", " + action)
            if clip[:-8] + ", " + action not in clip_action_label_train.keys():
                clip_action_label_train[clip[:-8] + ", " + action] = set()
            clip_action_label_train[clip[:-8] + ", " + action].add(label)
            set_miniclip_train.add(clip[:-8])
            set_clip_train.add(clip[:-4])
            set_videos_train.add(clip.split("mini")[0])

    for clip in list(dict_val_annotations.keys()):
        list_action_label = dict_val_annotations[clip]
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        # viz_feat = np.array(dict_miniclip_clip_feature[clip[:-4]])
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = dict_action_embeddings[action]

            data_clips_val.append([clip, viz_feat])
            data_actions_val.append([action, action_emb])
            labels_val.append(label)
            set_action_miniclip_val.add(clip[:-8] + ", " + action)
            if clip[:-8] + ", " + action not in clip_action_label_val.keys():
                clip_action_label_val[clip[:-8] + ", " + action] = set()
            clip_action_label_val[clip[:-8] + ", " + action].add(label)
            set_miniclip_val.add(clip[:-8])
            set_clip_val.add(clip[:-4])
            set_videos_val.add(clip.split("mini")[0])

    list_test_clip_names = []
    list_test_action_names = []
    for clip in list(dict_test_annotations.keys()):
        list_action_label = dict_test_annotations[clip]
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        # viz_feat = np.array(dict_miniclip_clip_feature[clip[:-4]])
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]

        for [action, label] in list_action_label:
            action_emb = dict_action_embeddings[action]

            data_clips_test.append([clip, viz_feat])
            data_actions_test.append([action, action_emb])
            labels_test.append(label)
            set_action_miniclip_test.add(clip[:-8] + ", " + action)
            if clip[:-8] + ", " + action not in clip_action_label_test.keys():
                clip_action_label_test[clip[:-8] + ", " + action] = set()
            clip_action_label_test[clip[:-8] + ", " + action].add(label)
            list_test_clip_names.append(clip)
            list_test_action_names.append(action)
            set_miniclip_test.add(clip[:-8])
            set_clip_test.add(clip[:-4])
            set_videos_test.add(clip.split("mini")[0])

    print(tabulate([['Total', 'Train', 'Val', 'Test'],
                    [len(data_actions_train) + len(data_actions_val) + len(data_actions_test), len(data_actions_train),
                     len(data_actions_val), len(data_actions_test)]],
                   headers="firstrow"))

    print(Counter(labels_train))
    print(Counter(labels_val))
    print(Counter(labels_test))

    print("# actions train " + str(len(set_action_miniclip_train)))
    print("# actions val " + str(len(set_action_miniclip_val)))
    print("# actions test " + str(len(set_action_miniclip_test)))

    print("# videos train " + str(len(set_videos_train)))
    print("# videos val " + str(len(set_videos_val)))
    print("# videos test " + str(len(set_videos_test)))

    print("# miniclips train " + str(len(set_miniclip_train)))
    print("# miniclips val " + str(len(set_miniclip_val)))
    print("# miniclips test " + str(len(set_miniclip_test)))

    print("# clips train " + str(len(set_clip_train)))
    print("# clips val " + str(len(set_clip_val)))
    print("# clips test " + str(len(set_clip_test)))

    return [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
           [data_clips_test, data_actions_test, labels_test]

'''
    Method
'''

# model similar to TALL (alignment score & regression is different + pre-trained model features used)
def create_MPU_model(input_dim_video, input_dim_text):

    action_input = tf.keras.layers.Input(shape=(input_dim_text,), name='action_input')
    action_output = tf.keras.layers.Dense(64)(action_input)
    action_output = tf.keras.layers.Dropout(0.4)(action_output)

    video_input = tf.keras.layers.Input(shape=(input_dim_video,), dtype='float32', name='video_input')
    video_output = tf.keras.layers.Dense(64)(video_input)
    video_output = tf.keras.layers.Dropout(0.4)(video_output)

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

    model = tf.keras.models.Model(inputs=[action_input, video_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def get_features_from_data(train_data, val_data, test_data):
    [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
    [data_clips_test, data_actions_test, labels_test] = train_data, val_data, test_data
    # features
    # data_clips_feat_train = np.asarray([i[1] for i in data_clips_train], dtype=object)
    data_clips_feat_train =[i[1] for i in data_clips_train]
    data_actions_emb_train = [i[1] for i in data_actions_train]

    # data_clips_feat_val = np.asarray([i[1] for i in data_clips_val], dtype=object)
    data_clips_feat_val = [i[1] for i in data_clips_val]
    data_actions_emb_val = [i[1] for i in data_actions_val]

    # data_clips_feat_test = np.asarray([i[1] for i in data_clips_test], dtype=object)
    data_clips_feat_test = [i[1] for i in data_clips_test]
    data_actions_emb_test = [i[1] for i in data_actions_test]

    return [data_clips_feat_train, data_actions_emb_train, labels_train], \
           [data_clips_feat_val, data_actions_emb_val, labels_val], \
           [data_clips_feat_test, data_actions_emb_test, labels_test]


def create_model(train_data, val_data, test_data, nb_epochs, balance, config_name):
    print("---------- Running " + config_name + " -------------------")

    [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val,
                                                                                     labels_val], \
    [data_clips_test, data_actions_test, labels_test] = get_features_from_data(train_data, val_data, test_data)
    input_dim_text = len(data_actions_train[0])
    input_dim_video = len(data_clips_train[0])
    print(input_dim_text, input_dim_video)

    model = create_MPU_model(input_dim_video, input_dim_text)

    if balance == True:
        file_path_best_model = 'data/model_checkpoints/' + config_name + '_balance' + '.hdf5'
    else:
        file_path_best_model = 'data/model_checkpoints/' + config_name + '_no_balance' + '.hdf5'

    checkpointer = ModelCheckpoint(monitor='val_accuracy',
                                   filepath=file_path_best_model,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=15)
    callback_list = [earlystopper, checkpointer]

    ##### for testing
    # data_actions_train, data_clips_train, data_actions_test, data_clips_test, data_clips_val,  data_actions_val, \
    # labels_train, labels_val, labels_test = data_actions_train[:100], data_clips_train[:100], data_actions_test[:100], data_clips_test[:100],\
    #                                         data_clips_val[:100], data_actions_val[:100], labels_train[:100], labels_val[:100], labels_test[:100]

    data_actions_train, data_clips_train, data_actions_test, data_clips_test, data_clips_val, data_actions_val, labels_train, labels_val, labels_test = \
        np.asarray(data_actions_train).astype(np.float32), np.asarray(data_clips_train).astype(np.float32), \
        np.asarray(data_actions_test).astype(np.float32), np.asarray(data_clips_test).astype(np.float32),\
        np.asarray(data_clips_val).astype(np.float32),  np.asarray(data_actions_val).astype(np.float32), \
        np.asarray(labels_train).astype(np.float32), np.asarray(labels_val).astype(np.float32), np.asarray(labels_test).astype(np.float32)

    if not os.path.isfile(file_path_best_model):
        model.fit([data_actions_train, data_clips_train], labels_train,
                  validation_data=([data_actions_val, data_clips_val], labels_val),
                  epochs=nb_epochs, batch_size=64, verbose=1, callbacks=callback_list)

    print("Load best model weights from " + file_path_best_model)
    model.load_weights(file_path_best_model)

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

    return predicted, list_predictions

'''
Evaluation
'''

def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]

    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.

    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return [list(a) for a in zip(*[lst[i::n] for i in range(n)])]


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
    # overlapping_sec = 10
    overlapping_sec = 3
    # print(groups)
    merged_intervals = merge_intervals(groups, overlapping_sec)

    merged_intervals.sort(key=lambda x: x[2], reverse=True)  # highest scored proposal
    # print(merged_intervals)
    proposal = [merged_intervals[0][:-1]]
    return proposal


def compute_predicted_IOU(model_name, predicted_labels_test, test_data, clip_length,
                          list_predictions):
    with open("data/dict_clip_time_per_miniclip" + clip_length + ".json") as f:
        dict_clip_time_per_miniclip = json.loads(f.read())

    [data_clips_test, data_actions_test, gt_labels_test] = test_data
    data_clips_test_names = [i[0] for i in data_clips_test]
    data_actions_test_names = [i[0] for i in data_actions_test]
    dict_predicted = {}

    data = zip(data_clips_test_names, data_actions_test_names, predicted_labels_test, list_predictions)
    output = "data/results/dict_predicted_" + model_name + ".json"
    counter_no_detected_action = 0
    counter_detected_action = 0
    for [clip, action, label, score] in data:
        miniclip = clip[:-8] + ".mp4"
        if label:
            if miniclip + ", " + action not in dict_predicted.keys():
                dict_predicted[miniclip + ", " + action] = []

            [time_s, time_e] = dict_clip_time_per_miniclip[clip]
            dict_predicted[miniclip + ", " + action].append(time_s)
            dict_predicted[miniclip + ", " + action].append(time_e)
            dict_predicted[miniclip + ", " + action].append(score)
        else:
            if miniclip + ", " + action not in dict_predicted.keys():
                dict_predicted[miniclip + ", " + action] = []

    for key in dict_predicted.keys():
        if not dict_predicted[key]:
            dict_predicted[key].append(-1)
            dict_predicted[key].append(-1)
            dict_predicted[key].append(-1)

            counter_no_detected_action += 1
        else:
            counter_detected_action += 1

    print("# no detected action in miniclip: " + str(counter_no_detected_action))
    print("# detected action in miniclip: " + str(counter_detected_action))

    for key in list(dict_predicted.keys()):
        list_all_times = dict_predicted[key]
        # print(key)
        proposal = compute_final_proposal(list_all_times)
        dict_predicted[key] = proposal

    with open(output, 'w+') as fp:
        json.dump(dict_predicted, fp)

def evaluate(method, channel):
    print("Results for method {0} on channel {1}:".format(method, channel))
    with open("data/results/dict_predicted_" + method + ".json") as f:
        proposed_1p0 = json.loads(f.read())

    with open("data/annotations/annotations" + channel + ".json") as f:
        groundtruth_1p0 = json.loads(f.read())


    IOU_vals, dict_IOU_per_length, dict_IOU_per_position = wrapper_IOU(proposed_1p0, groundtruth_1p0)
    print("#test points: " + str(len(IOU_vals)))

    list_results = []
    for threshold in np.arange(0.1, 0.9, 0.2):
        accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
        list_results.append(str(round(accuracy, 2)))

    mean_tIOU = compute_meanIOU(IOU_vals)
    list_results.append(str(round(mean_tIOU, 2)))
    print("overleaf: " + list_results[0] + " & " + list_results[1] + " & " + list_results[2] + " & " + list_results[3] + " & " + list_results[4])


def compute_accuracy_IOU_threshold(threshold, IOU_vals):
    '''
    In this setting, for a given
    value of threshold, whenever a given predicted time window has
    an intersection with the gold-standard that is above the α
    threshold, we consider the output of the model as correct.
    :param threshold: 0.1, 0.3, 0.5, 0.7
    '''
    nb_correct = 0
    nb_total = len(IOU_vals)
    for tIOU in IOU_vals:
        if max(tIOU) > threshold:
            nb_correct += 1
    tIOU_threshold = nb_correct / nb_total * 100
    print("tIOU@{:.1} = {:.2f}".format(threshold, tIOU_threshold))
    return tIOU_threshold

def compute_meanIOU(IOU_vals):
    mean_tIOU = np.nanmean([max(x) for x in IOU_vals])
    mean_tIOU = mean_tIOU * 100
    print("mIOU = {:.2f}".format(mean_tIOU))
    return mean_tIOU

def wrapper_IOU(proposed_1p0, groundtruth_1p0):
    if len(proposed_1p0.keys()) != len(groundtruth_1p0.keys()):
        count_visible_actions_not_caught = 0
        for key in groundtruth_1p0.keys() - proposed_1p0.keys():
            if groundtruth_1p0[key] != ['not visible']:
                count_visible_actions_not_caught += 1
                # print(key)
        if count_visible_actions_not_caught:
            print("count_visible_actions_not_caught: " + str(count_visible_actions_not_caught))

    IOU_vals = []
    dict_IOU_per_length = {}
    dict_IOU_per_position = {'10': [], '50': []}
    gt = len(groundtruth_1p0.keys())
    p = len(proposed_1p0.keys())
    print(gt)
    print(p)
    nb_not_vis_total = 0
    nb_not_vis = 0

    for miniclip_action in groundtruth_1p0.keys():
        if groundtruth_1p0[miniclip_action][0] == 'not visible':
            groundtruth_1p0[miniclip_action] = [-1, -1]
            nb_not_vis_total += 1
            # continue
        if miniclip_action not in proposed_1p0.keys():
            continue
        target_segment = np.array([float(x) for x in groundtruth_1p0[miniclip_action]])
        candidate_segments = np.array(proposed_1p0[miniclip_action])

        tIOU = segment_iou(target_segment, candidate_segments)
        if math.isnan(tIOU[0]):
            tIOU = [1]
            nb_not_vis += 1
            if groundtruth_1p0[miniclip_action] != [-1, -1] or proposed_1p0[miniclip_action][0] != [-1, -1]:
                print("NOOOO: ")
                print(groundtruth_1p0[miniclip_action])
                print(proposed_1p0[miniclip_action])
                break

        IOU_vals.append(tIOU)

        action_duration = target_segment[1] - target_segment[0]
        rounded_duration = str(int(round(action_duration, -1)))
        if rounded_duration == "10":
            rounded_duration = "0"
        elif rounded_duration == "30":
            rounded_duration = "20"
        elif rounded_duration in ["40", "50", "60"]:
            rounded_duration = "50"

        if rounded_duration not in dict_IOU_per_length.keys():
            dict_IOU_per_length[rounded_duration] = []
        dict_IOU_per_length[rounded_duration].append(tIOU)

        if target_segment[0] <= 10:
            dict_IOU_per_position["10"].append(tIOU)
        else:
            dict_IOU_per_position["50"].append(tIOU)

    print("# not visible predicted as not visible " + str(nb_not_vis) + "out of #GT " + str(nb_not_vis_total))
    return IOU_vals, dict_IOU_per_length, dict_IOU_per_position

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def wrapper_IOU_combine_2(predicted_time, proposed_1p0_1, proposed_1p0_2, groundtruth_1p0):

    if len(proposed_1p0_1.keys()) != len(groundtruth_1p0.keys()):
        count_visible_actions_not_caught = 0
        for key in groundtruth_1p0.keys() - proposed_1p0_1.keys():
            if groundtruth_1p0[key] != ['not visible']:
                count_visible_actions_not_caught += 1
                # print(key)
        if count_visible_actions_not_caught:
            print("count_visible_actions_not_caught: " + str(count_visible_actions_not_caught))

    if len(proposed_1p0_2.keys()) != len(groundtruth_1p0.keys()):
        count_visible_actions_not_caught = 0
        for key in groundtruth_1p0.keys() - proposed_1p0_2.keys():
            if groundtruth_1p0[key] != ['not visible']:
                count_visible_actions_not_caught += 1
                # print(key)
        if count_visible_actions_not_caught:
            print("count_visible_actions_not_caught: " + str(count_visible_actions_not_caught))

    IOU_vals = []
    dict_IOU_per_length = {}
    for miniclip_action in groundtruth_1p0.keys():
        # TODO: deal with this
        if groundtruth_1p0[miniclip_action] == ['not visible']:
            continue
        if miniclip_action not in proposed_1p0_1.keys():
            continue
        if proposed_1p0_1[miniclip_action] == ['not visible']:
            continue
        if miniclip_action not in proposed_1p0_2.keys():
            continue
        if proposed_1p0_2[miniclip_action] == ['not visible']:
            continue

        target_segment = np.array([float(x) for x in groundtruth_1p0[miniclip_action]])

        action_duration = target_segment[1] - target_segment[0]
        rounded_duration = str(int(round(action_duration, -1)))
        predicted_duration = int(predicted_time[miniclip_action.split(", ")[1]])

        if predicted_duration == 1:
            candidate_segments = np.array(proposed_1p0_1[miniclip_action])  # alignment is good for short actions
        else:
            candidate_segments = np.array(proposed_1p0_2[miniclip_action])  # MPU is good for long actions

        tIOU = segment_iou(target_segment, candidate_segments)
        IOU_vals.append(tIOU)

        if rounded_duration not in dict_IOU_per_length.keys():
            dict_IOU_per_length[rounded_duration] = []
        dict_IOU_per_length[rounded_duration].append(tIOU)

    return IOU_vals, dict_IOU_per_length

def evaluate_2SEAL(method1, method2, channel):
    print("Computing predicted action duration ...")
    # svm_predict_action_duration(channels_val, channels_test)
    with open("data/dict_predicted_time.json") as f:
        predicted_time = json.loads(f.read())

    print("Results for method {0}, {1} on channel {2}:".format(method1, method2, channel))
    with open("data/results/dict_predicted_" + method1 + ".json") as f:
        proposed_1p0_1 = json.loads(f.read())

    with open("data/results/dict_predicted_" + method2 + ".json") as f:
        proposed_1p0_2 = json.loads(f.read())

    with open("data/annotations/annotations" + channel + ".json") as f:
        groundtruth_1p0 = json.loads(f.read())

    IOU_vals, dict_IOU_per_length = wrapper_IOU_combine_2(predicted_time, proposed_1p0_1, proposed_1p0_2, groundtruth_1p0)
    print("#test points: " + str(len(IOU_vals)))

    list_results = []
    for threshold in np.arange(0.1, 0.9, 0.2):
        accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
        list_results.append(str(round(accuracy, 2)))

    mean_tIOU = compute_meanIOU(IOU_vals)
    list_results.append(str(round(mean_tIOU, 2)))
    print("overleaf: " + list_results[0] + " & " + list_results[1] + " & " + list_results[2] + " & " + list_results[3] + " & " + list_results[4])

    # for action_duration in dict_IOU_per_length.keys():
    #     IOU_vals = dict_IOU_per_length[action_duration]
    #     print(action_duration)
    #     print("#test points: " + str(len(IOU_vals)))
    #
    #     list_results = []
    #     for threshold in np.arange(0.1, 0.9, 0.2):
    #         accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
    #         list_results.append(str(round(accuracy, 2)))
    #
    #     mean_tIOU = compute_meanIOU(IOU_vals)
    #     list_results.append(str(round(mean_tIOU, 2)))
    #     print("overleaf: " + list_results[0] + " & " + list_results[1] + " & " + list_results[2] + " & " + list_results[3] + " & " + list_results[4])




'''
    Action Duration classification
'''

def get_extra_data_coin():
    X_train = []
    Y_train = []

    with open("data/embeddings/dict_action_embeddings_Bert_COIN2.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    with open("data/related_work/COIN.json") as file:
        coin_data = json.load(file)

    data = coin_data["database"]
    for key in data.keys():
        content = data[key]
        for i in range(len(content["annotation"])):
            action = content["annotation"][i]["label"]
            segment_time = content["annotation"][i]["segment"]
            action_duration = int(segment_time[1] - segment_time[0])
            action_emb = dict_action_embeddings_Bert[action]
            if action_duration in [0, 10]:
                action_duration = 1
            else:
                action_duration = 0
            X_train.append(action_emb)
            Y_train.append(action_duration)

    # downsample
    stop = 1
    while stop:
        for i, el in enumerate(Y_train):
            c = Counter(Y_train)
            if el == 0:
                del X_train[i]
                del Y_train[i]
            if c[1] >= c[0] * 3:
                stop = 0
                break

    return X_train, Y_train

def svm_predict_action_duration(channels_val, channels_test):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=0)

    with open("data/dict_all_annotations_1_10channels.json") as file:
        annotations = json.load(file)

    with open("data/embeddings/dict_action_embeddings_Bert2.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    # X_train, Y_train = get_extra_data_charades()
    X_train, Y_train = get_extra_data_coin()
    X_test, Y_test, X_test_action  = [], [], []


    for miniclip in annotations.keys():
        for [action, label] in annotations[miniclip]:
            if label != ['not visible']:
                [t_s_gt, t_e_gt] = label
                duration = int(round(t_e_gt - t_s_gt, -1))
                if duration in [0, 10]:
                    duration = 1  # short
                else:
                    duration = 0  # long
                emb_action_train = dict_action_embeddings_Bert[action]

                if miniclip.split("_")[0] in channels_test:
                    X_test.append(emb_action_train)
                    X_test_action.append(action)
                    Y_test.append(duration)
                else:
                    X_train.append(emb_action_train)
                    Y_train.append(duration)


    # Standarize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_train)

    print(Counter(Y_train))
    print(Counter(Y_test))

    print("Fitting model")
    model.fit(X_std, Y_train)

    print("Evaluating model test:")
    predicted_test = model.predict(X_test)

    acc_score = accuracy_score(Y_test, predicted_test)
    f1 = f1_score(Y_test, predicted_test)
    recall = recall_score(Y_test, predicted_test)
    precision = precision_score(Y_test, predicted_test)
    print("acc_score test: {:0.2f}".format(acc_score))
    print("f1_score test: {:0.2f}".format(f1))
    print("recall test: {:0.2f}".format(recall))
    print("precision test: {:0.2f}".format(precision))

    predicted_dict = {}
    for action, duration in zip(X_test_action, predicted_test):
        predicted_dict[action] = str(duration)

    with open("data/dict_predicted_time.json", 'w+') as fp:
        json.dump(predicted_dict, fp)


