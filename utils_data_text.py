from __future__ import print_function, absolute_import, unicode_literals, division

import itertools
import pickle
import random
import time

import scipy
from nltk import bigrams
import numpy as np
import json
from collections import OrderedDict

from sklearn import cluster
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import glob
import os
import re, math
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from keras.preprocessing.text import one_hot, Tokenizer
from transformers.tokenization_auto import AutoTokenizer
from transformers.modeling_bert import BertModel
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from compute_text_embeddings import embed_elmo2, get_bert_finetuned_embeddings, create_bert_embeddings, NumpyEncoder
from steve_human_action.main import read_activity, read_ouput_DNT
from utils_data_video import average_i3d_features, load_data_from_I3D, average_i3d_features_miniclip

from nltk.corpus import wordnet
from itertools import product

WORD = re.compile(r'\w+')

plt.style.use('ggplot')


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# def get_word_embedding(embeddings_index, word):
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is None:
#         return None
#     else:
#         word_embedding = np.asarray(embedding_vector)
#         return word_embedding
#
#
# def load_embeddings():
#     embeddings_index = dict()
#     with open("data/glove.6B.50d.txt") as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs
#     print('Loaded %s word vectors.' % len(embeddings_index))
#
#     return embeddings_index


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def verify_overlaps_actions(action1, action2):
    words_1 = action1.split()
    words_2 = action2.decode().split()
    for word in words_1:
        if word not in words_2 and stem_word(word) not in words_2:
            return 0
    return 1


'''
return -1, if time_1 < time_2
        1, if time_1 > time_2
        0, if equal
'''


def compare_time(time_1, time_2):
    minute_1_s, sec_1_s = time_1.split(":")
    minute_2_s, sec_2_s = time_2.split(":")

    if int(minute_1_s) == int(minute_2_s) and int(sec_1_s) == int(sec_2_s):
        return 0
    if int(minute_1_s) > int(minute_2_s) or (int(minute_1_s) == int(minute_2_s) and int(sec_1_s) >= int(sec_2_s)):
        return 1
    else:
        return -1


def get_time_difference(time_1, time_2):
    if compare_time(time_1, time_2) == -1:
        return 100
    FMT = '%M:%S'
    tdelta = datetime.strptime(time_1, FMT) - datetime.strptime(time_2, FMT)
    tdelta_h, tdelta_min, tdelta_seconds = str(tdelta).split(":")
    tdelta_total = int(tdelta_seconds) + int(tdelta_min) * 60 + int(tdelta_h) * 360
    return tdelta_total


def stem_word(word):
    handcraft_rules = {'ad': 'add', 'tri': 'try', 'saut': 'saute', 'danc': 'dance', 'remov': 'remove', 'has': 'have',
                       'did': 'do', 'made': 'make', 'cleans': 'clean', 'creat': 'create', 'dri': 'dry', 'done': 'do',
                       'ran': 'run', 'ate': 'eat', 'cleaning': 'clean', 'got': 'get',
                       'snuggl': 'snuggle', 'subscrib': 'subscribe', 'squeez': 'squeeze', 'chose': 'choose',
                       'bundl': 'bundle', 'decid': 'decide', 'empti': 'empty', 'wore': 'wear', 'starv': 'starve',
                       'increas': 'increase', 'incorpor': 'incorporate',
                       'purchas': 'purchase', 'laid': 'lay', 'rins': 'rinse', 'saw': 'see',
                       'goe': 'go', 'appli': 'apply', 'diffus': 'diffuse',
                       'combin': 'combine', 'shown': 'show', 'stapl': 'staple', 'burnt': 'burn', 'imagin': 'imagine',
                       'achiev': 'achieve', 'sped': 'speed', 'carri': 'carry',
                       'took': 'take', 'measur': 'measure', 'sprinkl': 'sprinkle', 'gave': 'give', 'thrown': 'throw',
                       'massag': 'massage',
                       'hydrat': 'hydrate', 'organ': 'organize', 'tidi': 'tidy', 'involv': 'involve', 'serv': 'serve',
                       'bought': 'buy',
                       'seen': 'see', 'prepar': 'prepare', 'went': 'go', 'exfoli': 'exfoliate', 'shelv': 'shelve',
                       'chosen': 'choose', 'assembl': 'assemble',
                       'inspir': 'inspire', 'kept': 'keep', 'complet': 'complete', 'fri': 'fry', 'infus': 'infuse',
                       'figur': 'figure', 'wast': 'waste', 'freez': 'freeze', 'simplifi': 'simplify',
                       'marin': 'marinate', 'consid': 'consider', 'written': 'write', 'colour': 'color',
                       'oppos': 'oppose', 'medit': 'meditate',
                       'met': 'meet', 'given': 'give', 'squegee': 'squege', 'flavour': 'flavor', 'gotten': 'got',
                       'came': 'come', 'knew': 'know',
                       'includ': 'include', 'stumbl': 'stumble', 'taken': 'take', 'consist': 'consists',
                       'penetr': 'penetrate', 'leav': 'leave', 'redecor': 'redecorate', 'tangl': 'tangle'}
    stemmer = SnowballStemmer("english")

    if word in handcraft_rules.keys():
        return handcraft_rules[word]

    if word[-1] == 'e' or word[-1] == 'y':
        return word

    word_stemmed = stemmer.stem(word)
    if word_stemmed in handcraft_rules.keys():
        word_stemmed = handcraft_rules[word_stemmed]

    return word_stemmed


def stem_action(action, path_pos_data):
    with open(path_pos_data) as f:
        dict_pos_actions = json.loads(f.read())

    if action in dict_pos_actions.keys():
        list_word_pos = dict_pos_actions[action]
        for [word, pos, concr_score] in list_word_pos:
            if 'VB' in pos:
                stemmed_word = stem_word(word)
                words = action.split()
                replaced = " ".join([stemmed_word if wd == word else wd for wd in words])
                action = replaced
    else:
        return None

    return action


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


def compute_predicted_IOU_GT(test_data, clip_length):
    with open("data/dict_clip_time_per_miniclip" + clip_length + ".json") as f:
        dict_clip_time_per_miniclip = json.loads(f.read())

    [data_clips_test, data_actions_test, gt_labels_test] = test_data
    data_clips_test_names = [i[0] for i in data_clips_test]
    data_actions_test_names = [i[0] for i in data_actions_test]
    dict_predicted = {}

    data = zip(data_clips_test_names, data_actions_test_names, gt_labels_test)
    output = "data/results/dict_predicted_GT.json"

    for [clip, action, label] in data:
        miniclip = clip[:-8] + ".mp4"
        if label:
            if miniclip + ", " + action not in dict_predicted.keys():
                dict_predicted[miniclip + ", " + action] = []

            [time_s, time_e] = dict_clip_time_per_miniclip[clip]
            dict_predicted[miniclip + ", " + action].append(time_s)
            dict_predicted[miniclip + ", " + action].append(time_e)
        else:
            if miniclip + ", " + action not in dict_predicted.keys():
                dict_predicted[miniclip + ", " + action] = []

    for key in dict_predicted.keys():
        if not dict_predicted[key]:
            dict_predicted[key].append(-1)
            dict_predicted[key].append(-1)

    for key in dict_predicted.keys():
        list_all_times = dict_predicted[key]
        dict_predicted[key] = [[min(list_all_times), max(list_all_times)]]

    with open(output, 'w+') as fp:
        json.dump(dict_predicted, fp)


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


def create_data_for_model(type_action_emb, balance, add_cluster, add_object_labels, add_object_feat,
                          path_all_annotations,
                          path_I3D_features, channels_val,
                          channels_test, hold_out_test_channels):
    with open(path_all_annotations) as f:
        dict_all_annotations = json.load(f)

    # # dict_miniclip_clip_feature = load_data_from_I3D(path_I3D_features) #if LSTM
    dict_miniclip_clip_feature = average_i3d_features(path_I3D_features)
    dict_miniclip_feature = average_i3d_features_miniclip("../i3d_keras/data/results_miniclip/")
    dict_action_embeddings = load_text_embeddings(type_action_emb, dict_all_annotations, all_actions=True,
                                                  use_nouns=True, use_particle=True)

    if add_cluster:
        dict_action_embeddings = add_cluster_data(dict_action_embeddings)

    if add_object_labels != "none":
        print("Adding object label data")
        dict_clip_object_labels = add_object_label(add_object_labels)
    else:
        dict_clip_object_labels = {}

    if add_object_feat != "none":
        print("Adding object feature data")
        dict_clip_object_features = add_object_features(add_object_feat)
    else:
        dict_clip_object_features = {}

    dict_train_annotations, dict_val_annotations, dict_test_annotations = split_data_train_val_test(
        dict_all_annotations,
        channels_val,
        channels_test,
        hold_out_test_channels)

    data_clips_train, data_actions_train, labels_train = [], [], []
    data_clips_val, data_actions_val, labels_val = [], [], []
    data_clips_test, data_actions_test, labels_test = [], [], []

    set_action_miniclip_train = set()
    set_action_miniclip_test = set()
    set_action_miniclip_val = set()

    set_videos_train = set()
    set_videos_test = set()
    set_videos_val = set()

    set_miniclip_train = set()
    set_miniclip_test = set()
    set_miniclip_val = set()

    set_clip_train = set()
    set_clip_test = set()
    set_clip_val = set()

    if balance:
        print("Balance data (train & val)")

        with open("data/train_test_val/dict_balanced_annotations_train_9_10.json") as f:
            dict_train_annotations = json.loads(f.read())
        # dict_train_annotations = balance_data(dict_train_annotations)
        # with open("data/train_test_val/dict_balanced_annotations_train_9_10.json", 'w+') as fp:
        #     json.dump(dict_train_annotations, fp)

        with open("data/train_test_val/dict_balanced_annotations_val_9_10.json") as f:
            dict_val_annotations = json.loads(f.read())
        # dict_val_annotations = balance_data(dict_val_annotations)
        # with open("data/train_test_val/dict_balanced_annotations_val_9_10.json", 'w+') as fp:
        #     json.dump(dict_val_annotations, fp)

        # No balancing the test!!!
        # with open("data/train_test_val/dict_balanced_annotations_test.json") as f:
        #     dict_test_annotations = json.loads(f.read())
        # dict_test_annotations = balance_data(dict_test_annotations)
        # with open("data/train_test_val/dict_balanced_annotations_test.json", 'w+') as fp:
        #     json.dump(dict_test_annotations, fp)

    for clip in list(dict_train_annotations.keys()):
        list_action_label = dict_train_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue
        viz_feat = dict_miniclip_clip_feature[clip[:-4]]
        # add or concat them
        if dict_clip_object_features != {}:
            viz_objects_feat = dict_clip_object_features[clip[:-4]]
            viz_feat = np.concatenate((viz_feat, viz_objects_feat), axis=0)


        miniclip_viz_feat = dict_miniclip_feature[clip[:-8]]
        pos_viz_feat = list(np.eye(1024)[int(clip[-7:-4])])
        viz_feat2 = [y for x in [viz_feat, miniclip_viz_feat] for y in x]
        viz_feat3 = [y for x in [viz_feat2, pos_viz_feat] for y in x]

        for [action, label] in list_action_label:
            # action, _ = compute_action(action, use_nouns=False, use_particle=True)
            action_emb = dict_action_embeddings[action]
            # TODO: concat
            if clip[:-4] in dict_clip_object_labels.keys():
                action_emb += dict_clip_object_labels[clip[:-4]]
                # action_emb = list(np.concatenate((np.array(action_emb), np.array(dict_clip_object_labels[clip[:-4]]))))
            # action_emb = np.zeros(1024)
            # data_clips_train.append([clip, viz_feat3])
            data_clips_train.append([clip, viz_feat])
            data_actions_train.append([action, action_emb])
            labels_train.append(label)
            set_action_miniclip_train.add(clip[:-8] + ", " + action)
            set_miniclip_train.add(clip[:-8])
            set_clip_train.add(clip[:-4])
            set_videos_train.add(clip.split("mini")[0])

    for clip in list(dict_val_annotations.keys()):
        list_action_label = dict_val_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue

        viz_feat = dict_miniclip_clip_feature[clip[:-4]]
        # add or concat them
        if dict_clip_object_features != {}:
            viz_objects_feat = dict_clip_object_features[clip[:-4]]
            viz_feat = np.concatenate((viz_feat, viz_objects_feat), axis=0)

        miniclip_viz_feat = dict_miniclip_feature[clip[:-8]]
        pos_viz_feat = list(np.eye(1024)[int(clip[-7:-4])])
        viz_feat2 = [y for x in [viz_feat, miniclip_viz_feat] for y in x]
        viz_feat3 = [y for x in [viz_feat2, pos_viz_feat] for y in x]

        for [action, label] in list_action_label:
            # action, _ = compute_action(action, use_nouns=False, use_particle=True)
            action_emb = dict_action_embeddings[action]
            if clip[:-4] in dict_clip_object_labels.keys():
                action_emb += dict_clip_object_labels[clip[:-4]]
                # action_emb = list(np.concatenate((np.array(action_emb), np.array(dict_clip_object_labels[clip[:-4]]))))
            # action_emb = np.zeros(1024)
            # data_clips_val.append([clip, viz_feat3])
            data_clips_val.append([clip, viz_feat])
            data_actions_val.append([action, action_emb])
            labels_val.append(label)
            set_action_miniclip_val.add(clip[:-8] + ", " + action)
            set_miniclip_val.add(clip[:-8])
            set_clip_val.add(clip[:-4])
            set_videos_val.add(clip.split("mini")[0])

    list_test_clip_names = []
    list_test_action_names = []
    for clip in list(dict_test_annotations.keys()):
        list_action_label = dict_test_annotations[clip]
        # TODO: Spliting the clips, these were extra or they were < 8s (could not run I3D on them) or truncated
        if clip[:-4] not in dict_miniclip_clip_feature.keys():
            continue

        viz_feat = dict_miniclip_clip_feature[clip[:-4]]
        # add or concat them
        if dict_clip_object_features != {}:
            viz_objects_feat = dict_clip_object_features[clip[:-4]]
            viz_feat = np.concatenate((viz_feat, viz_objects_feat), axis=0)

        miniclip_viz_feat = dict_miniclip_feature[clip[:-8]]
        pos_viz_feat = list(np.eye(1024)[int(clip[-7:-4])])
        viz_feat2 = [y for x in [viz_feat, miniclip_viz_feat] for y in x]
        viz_feat3 = [y for x in [viz_feat2, pos_viz_feat] for y in x]

        for [action, label] in list_action_label:
            # action, _ = compute_action(action, use_nouns=False, use_particle=True)
            action_emb = dict_action_embeddings[action]
            if clip[:-4] in dict_clip_object_labels.keys():
                action_emb += dict_clip_object_labels[clip[:-4]]
                # action_emb = list(np.concatenate((np.array(action_emb), np.array(dict_clip_object_labels[clip[:-4]]))))
            # action_emb = np.zeros(1024)
            # data_clips_test.append([clip, viz_feat3])
            data_clips_test.append([clip, viz_feat])
            data_actions_test.append([action, action_emb])
            labels_test.append(label)
            set_action_miniclip_test.add(clip[:-8] + ", " + action)
            list_test_clip_names.append(clip)
            list_test_action_names.append(action)
            set_miniclip_test.add(clip[:-8])
            set_clip_test.add(clip[:-4])
            set_videos_test.add(clip.split("mini")[0])

    # np.save("data/clip_names1p1.npy", list_test_clip_names)
    # np.save("data/action_names1p1.npy", list_test_action_names)
    # np.save("data/labels1p1.npy", labels_test)

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

    # return [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
    #        [data_clips_test, data_actions_test, labels_test]


def load_text_embeddings(type_action_emb, dict_all_annotations, all_actions, use_nouns, use_particle):
    set_actions = set()
    for clip in list(dict_all_annotations.keys()):
        list_action_label = dict_all_annotations[clip]
        for [action, _] in list_action_label:
            if not all_actions:
                action, _ = compute_action(action, use_nouns, use_particle)
            set_actions.add(action)
    list_all_actions = list(set_actions)

    # with open("data/annotations/annotations1p01_5p01.json") as f:
    #     groundtruth_1p0 = json.loads(f.read())

    # GT_vb_noun = {}
    # for miniclip_action in groundtruth_1p0.keys():
    #     action = miniclip_action.split(", ")[1]
    #     miniclip = miniclip_action.split(", ")[0]
    #     new_action, _ = compute_action(action, use_nouns, use_particle)
    #     GT_vb_noun[miniclip + ", " + new_action] = groundtruth_1p0[miniclip_action]
    # with open('data/annotations/annotations1p01_5p01_vb.json', 'w+') as outfile:
    #     json.dump(GT_vb_noun, outfile, cls=NumpyEncoder)
    #
    # if type_action_emb == "GloVe":
    #     return create_glove_embeddings(list_all_actions)
    if type_action_emb == "ELMo":
        with open('data/embeddings/dict_action_embeddings_ELMo.json') as f:
            # with open('data/embeddings/dict_action_embeddings_ELMo_vb_particle.json') as f:
            json_load = json.loads(f.read())
        return json_load
        # return save_elmo_embddings(list_all_actions)  # if need to create new
    elif type_action_emb == "Bert":
        with open('data/embeddings/dict_action_embeddings_Bert2.json') as f:
            # with open('data/embeddings/dict_action_embeddings_Bert_only_vb.json') as f:
            json_load = json.loads(f.read())
        return json_load
        # return create_bert_embeddings(list_all_actions, path_output)
    elif type_action_emb == "DNT":
        dict_embeddings = read_ouput_DNT()
        dict_actions = read_activity()

        dict_action_emb = {}

        for index in dict_embeddings.keys():
            action = dict_actions[index]
            emb = dict_embeddings[index]
            dict_action_emb[action] = emb.reshape(-1)
        return dict_action_emb

        # with open('steve_human_action/dict_action_emb_DNT.json') as f:
        #     json_load = json.loads(f.read())
        # return json_load
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


def compute_median_per_miniclip(data_actions_names_test, data_clips_names_test, predicted, labels_test,
                                med_filt_kernel_size):
    dict_order_by_action = OrderedDict()
    for (action, clip, p, gt) in list(zip(data_actions_names_test, data_clips_names_test, predicted, labels_test)):
        # dict_order_by_action[(action, clip)] = [p[0], gt]
        dict_order_by_action[(action, clip)] = [p, gt]

    dict_pred_before_med = OrderedDict()
    for key in sorted(dict_order_by_action.keys()):
        action = key[0]
        clip = key[1]

        new_key = (action, "_".join(clip.split("_")[:-1]))
        if new_key not in dict_pred_before_med.keys():
            dict_pred_before_med[new_key] = []
        dict_pred_before_med[new_key].append(dict_order_by_action[key])

    dict_pred_after_med = OrderedDict()

    # for med_filt_kernel_size in [7, 11, 13, 15, 17, 21]:
    #     sum_bef = 0
    #     sum_aft = 0
    #     sum_bef_1 = 0
    #     sum_aft_1 = 0
    #     print("med_filt_kernel_size: " + str(med_filt_kernel_size))

    for key in dict_pred_before_med.keys():
        predicted_by_action_miniclip = [i[0] for i in dict_pred_before_med[key]]
        # print("predicted_by_action_miniclip:")
        # print(predicted_by_action_miniclip)

        GT_by_action_miniclip = [i[1] for i in dict_pred_before_med[key]]
        # print("GT_by_action_miniclip:")
        # print(GT_by_action_miniclip)
        acc_test_before = accuracy_score(GT_by_action_miniclip, predicted_by_action_miniclip)
        f1_test_before = f1_score(GT_by_action_miniclip, predicted_by_action_miniclip)
        # print("acc_test_before: " + str(acc_test_before))

        predicted_after_med = scipy.signal.medfilt(predicted_by_action_miniclip, med_filt_kernel_size)
        for v in range(len(predicted_after_med)):
            key1 = key[1] + "_{0:03}.mp4".format(v + 1)
            final_key = (key1, key[0])
            dict_pred_after_med[final_key] = predicted_after_med[v]
        # print("predicted_after_med:")
        # print(predicted_after_med)
        acc_test_after = accuracy_score(GT_by_action_miniclip, predicted_after_med)
        f1_test_after = f1_score(GT_by_action_miniclip, predicted_after_med)
        # print("acc_test_after: " + str(acc_test_after))

    # for key in sorted(dict_pred_after_med.keys())[0:10]:
    #     print(key, "::", str(dict_pred_after_med[key]))

    med_filt_predicted = []
    for (action, clip, p, gt) in list(zip(data_actions_names_test, data_clips_names_test, predicted, labels_test)):
        if (clip, action) not in dict_pred_after_med.keys():
            med_filt_predicted.append(False)
            continue
        med_filt_predicted.append(dict_pred_after_med[(clip, action)])
    return med_filt_predicted


def predict_action_duration(channels_test):
    with open("data/dict_all_annotations_1_10channels.json") as file:
        annotations = json.load(file)

    with open("data/embeddings/dict_action_embeddings_Bert.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    # with open("data/embeddings/dict_action_embeddings_Bert_vb_particle_noun.json") as f:
    #     dict_action_embeddings_Bert = json.loads(f.read())

    # dict_embeddings = read_ouput_DNT()
    # dict_actions = read_activity()
    #
    # dict_action_embeddings_Bert = {}
    #
    # for index in dict_embeddings.keys():
    #     action = dict_actions[index]
    #     emb = dict_embeddings[index]
    #     dict_action_embeddings_Bert[action] = emb.reshape(-1)

    # show actions grouped by GT duration - only vb
    list_test = []
    list_train = []
    for miniclip in annotations.keys():
        for [action, label] in annotations[miniclip]:
            if label != ['not visible']:
                [t_s_gt, t_e_gt] = label
                duration = int(round(t_e_gt - t_s_gt, -1))
                if duration in [0, 10]:
                    duration = 0  # short
                else:
                    duration = 1  # long
                if miniclip.split("_")[0] in channels_test:
                    list_test.append((action, duration))
                else:
                    list_train.append((action, duration))

    predicted_time = []
    GT_time = []
    for (action_test, time_gt) in tqdm(list_test):
        # action_test = action_test.split()[0]
        # action_test, _ = compute_action(action_test, use_nouns=True, use_particle=True)
        emb_action_test = dict_action_embeddings_Bert[action_test]
        max_score = 0
        max_time = 0
        max_action = ""

        for (action_train, time) in list_train:
            # action_train = action_train.split()[0]
            # action_train, _ = compute_action(action_train, use_nouns=True, use_particle=True)
            emb_action_train = dict_action_embeddings_Bert[action_train]
            cosine = scipy.spatial.distance.cosine(emb_action_test, emb_action_train)
            score = (1 - cosine) * 100
            if score > max_score:
                max_score = score
                max_time = time
                max_action = action_train
            if score >= 99:
                break

        predicted_time.append(max_time)
        GT_time.append(time_gt)
        # print(action_test + " = " + max_action)
    acc_score = accuracy_score(GT_time, predicted_time)
    f1 = f1_score(GT_time, predicted_time)
    recall = recall_score(GT_time, predicted_time)
    precision = precision_score(GT_time, predicted_time)
    print(
        "acc_score: {:0.2f}".format(acc_score))  # 0.66 - whole action; 0.64 - vb+particle+noun 0.63 - DNT Whole action
    print("f1_score: {:0.2f}".format(f1))  # 0.34 - only verb 0.38 - Bert whole action
    print("recall: {:0.2f}".format(recall))
    print("precision: {:0.2f}".format(precision))

    predicted_dict = {}
    for action, duration in zip(GT_time, predicted_time):
        predicted_dict[action] = duration


def get_extra_data_charades():
    list_all_actions = set()
    X_train = []
    Y_train = []

    with open("data/embeddings/dict_action_embeddings_Bert_Charades.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    with open("data/RelatedWorkDatasets/charadesSTA_test") as file_in:
        # actions_length = []
        for line in file_in:
            s = float(line.split(" ")[1])
            e = float(line.split(" ")[2])
            action = " ".join(line.split(" ")[4:])[:-2]
            list_all_actions.add(action)
            action_duration = e - s
            action_emb = dict_action_embeddings_Bert[action]
            # rounded_duration = int(round(action_duration, -1))
            if action_duration in [0, 10]:
                action_duration = 0
            else:
                action_duration = 1
            X_train.append(action_emb)
            Y_train.append(action_duration)

    with open("data/RelatedWorkDatasets/charadesSTA_train.txt") as file_in:
        # actions_length = []
        for line in file_in:
            s = float(line.split(" ")[1])
            e = float(line.split(" ")[2])
            action_duration = e - s
            action = " ".join(line.split(" ")[4:])[:-2]
            list_all_actions.add(action)
            action_emb = dict_action_embeddings_Bert[action]
            # rounded_duration = int(round(action_duration, -1))
            if action_duration in [0, 10]:
                action_duration = 0
            else:
                action_duration = 1
            X_train.append(action_emb)
            Y_train.append(action_duration)
            # print(action , str(action_duration))

    # create_bert_embeddings(list_all_actions)
    return X_train, Y_train


def get_extra_data_coin():
    list_all_actions = set()
    X_train = []
    Y_train = []

    with open("data/embeddings/dict_action_embeddings_Bert_COIN2.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    with open("data/RelatedWorkDatasets/COIN.json") as file:
        coin_data = json.load(file)

    data = coin_data["database"]
    # set_actions = set()
    for key in data.keys():
        content = data[key]
        for i in range(len(content["annotation"])):
            action = content["annotation"][i]["label"]
            # set_actions.add(action)
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
    # create_bert_embeddings(list(set_actions))


def svm_predict_actions(channels_test):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', class_weight='balanced', C=1.0, random_state=0)

    with open("data/dict_all_annotations_1_10channels.json") as file:
        annotations = json.load(file)

    with open("data/embeddings/dict_action_embeddings_Bert2.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    # with open("data/embeddings/dict_action_embeddings_Bert_vb_particle_noun.json") as f:
    #     dict_action_embeddings_Bert = json.loads(f.read())

    # dict_embeddings = read_ouput_DNT()
    # dict_actions = read_activity()
    #
    # dict_action_embeddings_Bert = {}
    #
    # for index in dict_embeddings.keys():
    #     action = dict_actions[index]
    #     emb = dict_embeddings[index]
    #     dict_action_embeddings_Bert[action] = emb.reshape(-1)
    #

    # X_train, Y_train = get_extra_data_charades()
    X_train, Y_train = get_extra_data_coin()
    #
    # print(Counter(Y_train))

    # X_train, Y_train = [], []
    X_test = []
    Y_test = []
    X_test_action = []
    X_train_action = []
    X_train_channel = []
    X_test_channel = []

    for miniclip in annotations.keys():
        # channel = miniclip.split("_")[0].split("p")[0]
        # channel_emb = np.ones(3) * int(channel)

        for [action, label] in annotations[miniclip]:
            if label != ['not visible']:
                [t_s_gt, t_e_gt] = label
                duration = int(round(t_e_gt - t_s_gt, -1))
                if duration in [0, 10]:
                    duration = 1  # short
                else:
                    duration = 0  # long
                # action_2, _ = compute_action(action, use_nouns=True, use_particle=True)
                # emb_action_train = dict_action_embeddings_Bert[action_2]
                emb_action_train = dict_action_embeddings_Bert[action]
                # emb_action_train = np.concatenate((emb_action_train, channel_emb), axis=0)
                # print(emb_action_train)
                if miniclip.split("_")[0] in channels_test:
                    X_test.append(emb_action_train)
                    X_test_action.append(action)
                    # X_test_channel.append(channel)
                    Y_test.append(duration)
                else:
                    X_train.append(emb_action_train)
                    Y_train.append(duration)
                    # X_train_channel.append(channel)
                    # X_train_action.append(action)

    # Standarize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_train)

    print(Counter(Y_train))
    print(Counter(Y_test))

    print("Fitting model")
    model.fit(X_std, Y_train)
    print("Evaluating model")
    predicted = model.predict(X_test)

    acc_score = accuracy_score(Y_test, predicted)
    f1 = f1_score(Y_test, predicted)
    recall = recall_score(Y_test, predicted)
    precision = precision_score(Y_test, predicted)
    print(
        "acc_score: {:0.2f}".format(acc_score))  # 0.66 - whole action; 0.64 - vb+particle+noun 0.63 - DNT Whole action
    print("f1_score: {:0.2f}".format(f1))  # 0.34 - only verb; 0.47 - Bert
    print("recall: {:0.2f}".format(recall))  # 0.34 - only verb; 0.47 - Bert
    print("precision: {:0.2f}".format(precision))  # 0.34 - only verb; 0.47 - Bert

    predicted_dict = {}
    for action, duration in zip(X_test_action, predicted):
        predicted_dict[action] = str(duration)

    # index = 0
    # for action, duration, gt in list(zip(X_test_action, predicted, Y_test)):
    #     if "make" in action and gt == 0:
    #         print("short: " + action)
    #     if "make" in action and gt == 1:
    #         print("long: " + action)

    # for verb in ['take','put','use','add','get','sprinkle','go','mix','cut','do']:
    #     if verb in action.split()[0]:
    #         print(verb, action.split()[0], action, str(gt), str(predicted[index]))
    #         predicted[index] = 0
    #         break
    # for verb in ['wipe','clean','make','prep','eat','write','vacuum','tidy','mop','cook','watch']:
    #     if verb in action.split()[0]:
    #         print(verb, action.split()[0], action, str(gt), str(predicted[index]))
    #         predicted[index] = 1
    #         break

    #     index += 1
    #
    # f1 = f1_score(Y_test, predicted)
    # recall = recall_score(Y_test, predicted)
    # precision = precision_score(Y_test, predicted)
    # print("acc_score: {:0.2f}".format(acc_score))  # 0.66 - whole action; 0.64 - vb+particle+noun 0.63 - DNT Whole action
    # print("f1_score: {:0.2f}".format(f1))  # 0.34 - only verb; 0.47 - Bert
    # print("recall: {:0.2f}".format(recall))  # 0.34 - only verb; 0.47 - Bert
    # print("precision: {:0.2f}".format(precision))  # 0.34 - only verb; 0.47 - Bert
    #
    # predicted_dict = {}
    # for action, duration in zip(X_test_action, predicted):
    #     predicted_dict[action] = duration

    return predicted_dict


def compute_predicted_IOU(model_name, predicted_labels_test, test_data, clip_length,
                          list_predictions):
    with open("data/dict_clip_time_per_miniclip" + clip_length + ".json") as f:
        dict_clip_time_per_miniclip = json.loads(f.read())

    with open("data/dict_time_per_miniclip.json") as f:
        dict_time_per_miniclip = json.loads(f.read())

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
            # miniclip_length = dict_time_per_miniclip[key.split(", ")[0]]
            # ## we don't know where, but we know the action is somewhere
            # dict_predicted[key].append(1)
            # # dict_predicted[key].append(1)
            # dict_predicted[key].append(miniclip_length)
            # dict_predicted[key].append(1)

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


def stemm_list_actions(list_actions, path_pos_data):
    stemmed_actions = []
    for action in list_actions:
        stemmed_action = stem_action(action, path_pos_data)
        if stemmed_action:
            stemmed_actions.append(stemmed_action)
        else:
            print(action + "not in dict_pos_action!")
    return stemmed_actions


def map_old_to_new_urls(path_old_urls, path_new_urls):
    list_old_csv_files = sorted(glob.glob(path_old_urls + "*.csv"))
    list_new_csv_files = sorted(glob.glob(path_new_urls + "*.csv"))

    for index in range(0, len(list_new_csv_files)):
        path_old_url = list_old_csv_files[index]
        path_new_url = list_new_csv_files[index]
        name_file = path_new_url.split("/")[-1]

        df_old = pd.read_csv(path_old_url)
        list_old_urls = list(df_old["Video_URL"])

        dict_new_urls = pd.read_csv(path_new_url, index_col=0, squeeze=True).to_dict()

        list_new_file = []
        for url_old in list_old_urls:
            if url_old in dict_new_urls.keys():
                list_new_file.append([url_old, dict_new_urls[url_old]])
            else:
                list_new_file.append([url_old, ''])
        df = pd.DataFrame(list_new_file)
        # df = df.transpose()
        df.columns = ["Video_URL", "Video_Name"]
        df.to_csv('data/new_video_urls/' + name_file, index=False)


def measure_nb_words(list_actions):
    dict_nb_words = {}
    for action in list_actions:
        nb_words = len(action.split(" "))
        if nb_words not in dict_nb_words.keys():
            dict_nb_words[nb_words] = 0
        dict_nb_words[nb_words] += 1

    plt.bar(list(dict_nb_words.keys()), dict_nb_words.values(), color='g')
    plt.xlabel("# words")
    plt.ylabel("# actions")
    plt.title("Number of words per NOT visible action")
    # plt.show()
    plt.savefig('data/stats/nb_words_per_NOT_visibile_action.png')


def measure_verb_distribution(list_actions, path_pos_data):
    dict_verbs = {}
    dict_stemmed_verbs = {}

    with open(path_pos_data) as f:
        dict_pos_actions = json.loads(f.read())

    for action in dict_pos_actions.keys():
        if action in list_actions:
            list_word_pos = dict_pos_actions[action]
            for [word, pos, concr_score] in list_word_pos:
                if 'VB' in pos:
                    if word not in dict_verbs.keys():
                        dict_verbs[word] = 0
                    dict_verbs[word] += 1

                    stemmed_word = stem_word(word)
                    if stemmed_word not in dict_stemmed_verbs.keys():
                        dict_stemmed_verbs[stemmed_word] = 0
                    dict_stemmed_verbs[stemmed_word] += 1

    sorted_dict = OrderedDict(sorted(dict_verbs.items(), key=lambda t: t[1], reverse=True))
    sorted_stemmed_dict = OrderedDict(sorted(dict_stemmed_verbs.items(), key=lambda t: t[1], reverse=True))

    print("Nb of different verbs before stemming: " + str(len(sorted_dict.keys())))
    print("Nb of different verbs after stemming: " + str(len(sorted_stemmed_dict.keys())))

    # df = pd.DataFrame(list(sorted_stemmed_dict.keys()))
    # df.columns = ["Unique Visible Actions Verbs after Stemming"]
    # df.to_csv('data/list_stemmed_verbs_unique_visibile.csv', index=False)

    # index = 0
    # for key in sorted_stemmed_dict:
    #     print(key, sorted_stemmed_dict[key])
    #     index += 1
    #     if index == 30:
    #         break
    # print("---------------------------------------")
    #
    plt.bar(list(sorted_stemmed_dict.keys())[:10], list(sorted_stemmed_dict.values())[:10], color='b')
    plt.xlabel("verb")
    plt.ylabel("# actions than contain the verb")
    plt.title("First 10 diff verbs for visibile actions actions")
    # plt.show()
    # plt.savefig('data/stats/first_10_verbs_visible_actions.png')


def analyze_verbs(list_verbs, action_list):
    list_all = []

    for verb in list_verbs:
        list_actions = []
        for action in action_list:
            if verb in action.split():
                list_actions.append(action)
        unique_actions = list(set(list_actions))

        print("There are " + str(len(list_actions)) + " actions with " + color.PURPLE + color.BOLD + verb + color.END)
        print("There are " + str(
            len(unique_actions)) + " unique actions with " + color.PURPLE + color.BOLD + verb + color.END)

        list_non_overlapping_actions = set()
        for i in range(0, len(unique_actions)):
            action_1 = unique_actions[i]
            ok = 1
            for j in range(0, len(unique_actions)):
                action_2 = unique_actions[j]
                if action_1 != action_2 and action_1 in action_2:
                    # print(action_1 + " | from | " + action_2)
                    ok = 0
                    break

            if ok == 1:
                list_non_overlapping_actions.add(action_1)
                if i == len(unique_actions) - 2:
                    list_non_overlapping_actions.add(unique_actions[-1])

        print("There are " + str(len(list(
            list_non_overlapping_actions))) + " unique non-overlapping actions with " + color.PURPLE + color.BOLD + verb + color.END)
        print("--------------------------------------------------------------------")
        # for action in list_non_overlapping_actions:
        #     print(action)

        list_all.append(list(list_non_overlapping_actions))

    # df = pd.DataFrame(list_all)
    # df = df.transpose()
    # df.columns = list_verbs
    # df.to_csv('data/list_TOP_VERBS_visible.csv', index=False)


def stats(list_actions, path_pos_data):
    # measure_nb_words(list_actions)
    measure_verb_distribution(list_actions, path_pos_data)


def write_list_to_csv(list_miniclips_visibile, list_visible_actions, list_stemmed_visibile_actions,
                      list_miniclips_not_visibile, list_not_visible_actions, list_stemmed_not_visibile_actions):
    list_all = []
    list_all.append(list_miniclips_visibile)
    list_all.append(list_visible_actions)
    list_all.append(list_stemmed_visibile_actions)
    list_all.append(list_miniclips_not_visibile)
    list_all.append(list_not_visible_actions)
    list_all.append(list_stemmed_not_visibile_actions)

    df = pd.DataFrame(list_all)
    df = df.transpose()
    df.columns = ["Miniclip Visible", "Visible Actions", "Stemmed Visible Actions", "Miniclip NOT Visible",
                  "NOT Visible Actions",
                  "Stemmed NOT Visible Actions"]
    df.to_csv('data/stats/list_actions.csv', index=False)


def separate_visibile_actions(path_miniclips, video):
    visible_actions = []
    not_visible_actions = []
    list_miniclips_visibile = []
    list_miniclips_not_visibile = []

    with open(path_miniclips) as f:
        dict_video_actions = json.loads(f.read())

    for miniclip in dict_video_actions.keys():

        if video != None:
            if video not in miniclip:
                continue

        list_actions_labels = dict_video_actions[miniclip]
        for [action, label] in list_actions_labels:
            if label == 0:
                visible_actions.append(action)
                list_miniclips_visibile.append(miniclip)
            else:
                not_visible_actions.append(action)
                list_miniclips_not_visibile.append(miniclip)
    return list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions


def time_flow_actions(path_miniclips, visible=True):
    if visible:
        name_column_miniclip = 'Miniclip Visible'
        name_column_actions = 'Stemmed Visible Actions'
        path_csv_output = 'data/stats/list_actions_video_ordered_stemmed_visible.csv'
    else:
        name_column_miniclip = 'Miniclip NOT Visible'
        name_column_actions = 'Stemmed NOT Visible Actions'
        path_csv_output = 'data/stats/list_actions_video_ordered_stemmed_NOT_visible.csv'

    df = pd.read_csv(path_miniclips)
    df = df.dropna()
    list_miniclips = df[name_column_miniclip]
    list_stemmed_actions = df[name_column_actions]

    dict_miniclip_actions = {}
    index = 0
    for miniclip in list_miniclips:
        if str(miniclip) == 'nan':
            continue
        if miniclip not in dict_miniclip_actions.keys():
            dict_miniclip_actions[miniclip] = []
        dict_miniclip_actions[miniclip].append(list_stemmed_actions[index])
        index += 1

    # order miniclips
    set_video_full_names = set()
    for miniclip in list_miniclips:
        channel_playlists, video, miniclip_id = miniclip.split("_")
        video = video
        miniclip_id = miniclip_id[:-4]

        video_full_name = channel_playlists + "_" + video
        set_video_full_names.add(video_full_name)

    list_video_full_names = list(set_video_full_names)
    print("There are " + str(len(list_video_full_names)) + " videos")
    dict_video = {}
    for video_name in list_video_full_names:
        set_miniclip_names = set()
        for miniclip_name in list_miniclips:
            if video_name in miniclip_name:
                set_miniclip_names.add(miniclip_name)

        list_miniclip_names = list(set_miniclip_names)
        # sort list of miniclips
        list_1 = []
        list_2 = []
        for miniclip_name in list_miniclip_names:
            if int(miniclip_name.split("_")[2][:-4]) > 9:
                list_2.append(miniclip_name)
            else:
                list_1.append(miniclip_name)
        sorted_list_miniclip_names = sorted(list_1) + sorted(list_2)
        dict_video[video_name] = sorted_list_miniclip_names

    list_all = []
    dict_video_actions = {}
    for video_name in dict_video.keys():
        list_ordered_actions = []
        for miniclip in dict_video[video_name]:
            list_ordered_actions += dict_miniclip_actions[miniclip]
        dict_video_actions[video_name] = list_ordered_actions
        list_all.append([video_name] + list_ordered_actions)

    df = pd.DataFrame(list_all)
    df = df.transpose()

    df.to_csv(path_csv_output, index=False, header=0)

    return dict_miniclip_actions, dict_video_actions


def sort_dict_pos():
    path_pos_data = "data/dict_action_pos_concreteness.json"
    sorted_dict_pos = {}

    with open(path_pos_data) as f:
        dict_pos_actions = json.loads(f.read())

    for action in dict_pos_actions.keys():
        list_words = action.split()
        sorted_dict_pos[action] = []
        for word in list_words:
            for [word_1, pos_1, score_1] in dict_pos_actions[action]:
                if word == word_1:
                    sorted_dict_pos[action].append([word_1, pos_1, score_1])
                    break

    with open('data/dict_action_pos_concreteness.json', 'w+') as outfile:
        json.dump(sorted_dict_pos, outfile)


def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Create bigrams from all words in corpus
    bi_grams = list(bigrams(corpus))

    # Frequency distribution of bigrams ((word1, word2), num_occurrences)
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))

    # Initialise co-occurrence matrix
    # co_occurrence_matrix[current][previous]
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))

    # Loop through the bigrams taking the current and previous word,
    # and the number of occurrences of the bigram.
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)

    # return the matrix and the index
    return co_occurrence_matrix, vocab_index


def sort_matrix_by_nb_coocurence(matrix, vocab_index):
    matrix = np.squeeze(np.asarray(matrix))
    sorted_indices_row = np.argsort(matrix.sum(axis=1))[
                         ::-1]  # sort indices by row (sum of the elemnts) & reverse list (to be descending order)
    sorted_matrix = matrix[:, sorted_indices_row]

    sorted_indices_col = np.argsort(sorted_matrix.sum(axis=0))[
                         ::-1]  # sort indices by column (sum of the elemnts) & reverse list (to be descending order)
    sorted_matrix = sorted_matrix[:, sorted_indices_col]

    sorted_indices_row_col = [0] * len(sorted_indices_row)
    for i in range(len(sorted_indices_row)):
        pos = sorted_indices_col[i]
        sorted_indices_row_col[i] = sorted_indices_row[pos]

    return sorted_indices_row_col, sorted_matrix


# TODO: remove stop words
def test_cooccurence_matrix(text_data):
    text_data = [t.split(",") for t in text_data]

    # Create one list using many lists
    data = list(itertools.chain.from_iterable(text_data))
    matrix, vocab_index = generate_co_occurrence_matrix(data)
    sorted_indices, sorted_matrix = sort_matrix_by_nb_coocurence(matrix, vocab_index)
    sorted_vocab_index = {}

    for k in vocab_index.keys():
        sorted_vocab_index[k] = sorted_indices.index(vocab_index[k])

    sorted_vocab_index_v = [v for (v, _) in sorted(sorted_vocab_index.items(), key=lambda kv: kv[1])]

    data_matrix = pd.DataFrame(matrix, index=vocab_index,
                               columns=vocab_index)

    data_matrix_sorted = pd.DataFrame(sorted_matrix, index=sorted_vocab_index,
                                      columns=sorted_vocab_index_v)

    data_matrix = data_matrix.drop(columns=['END'])
    data_matrix = data_matrix.drop(['END'])
    data_matrix = data_matrix[(data_matrix.T != 0).any()]
    data_matrix[data_matrix == 0] = np.nan

    data_matrix_sorted = data_matrix_sorted.drop(columns=['END'])
    data_matrix_sorted = data_matrix_sorted.drop(['END'])
    data_matrix_sorted = data_matrix_sorted[(data_matrix_sorted.T != 0).any()]
    data_matrix_sorted[data_matrix_sorted == 0] = np.nan

    column_values = list(data_matrix_sorted.columns.values)
    # print(column_values[0:5])
    # maxValuesObj = data_matrix_sorted.max()
    # print(maxValuesObj)
    first_data_matrix = data_matrix_sorted.nlargest(30, column_values[0])  # sort the rows
    first_data_matrix = first_data_matrix.iloc[:, :30]  # first 50 columns

    # first_data_matrix = data_matrix_sorted.iloc[:, :50]  # first 50 columns
    # first_data_matrix = first_data_matrix.iloc[:50, :]  # first 50 rows

    cmap = sns.cm.rocket_r

    # fig, axs = plt.subplots(ncols=2)
    # # sns.heatmap(data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
    # #             linewidths=1, ax = axs[0])
    #
    # sns.heatmap(data_matrix_sorted, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
    #             linewidths=1, ax = axs[0])
    #
    # sns.heatmap(first_data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
    #             linewidths=1, ax=axs[1])

    sns.heatmap(first_data_matrix, annot=True, cmap=cmap, xticklabels=1, yticklabels=1, cbar=True, square=True,
                linewidths=1, )

    # sns.clustermap(data_matrix, xticklabels=1, yticklabels=1,cmap=cmap)
    plt.show()


def write_mini_action_time_to_csv(list_miniclips_visibile, list_stemmed_visibile_actions,
                                  list_miniclips_not_visibile, list_stemmed_not_visibile_actions):
    list_all = []
    list_all.append(list_miniclips_visibile)
    list_all.append(list_stemmed_visibile_actions)
    list_all.append(list_miniclips_not_visibile)
    list_all.append(list_stemmed_not_visibile_actions)

    df = pd.DataFrame(list_all)
    df = df.transpose()
    df.columns = ["Miniclip Visible", "Stemmed Visible Actions", "Miniclip NOT Visible",
                  "Stemmed NOT Visible Actions"]
    df.to_csv('data/stats/list_actions_time.csv', index=False)


def write_json_for_annotations(list_miniclips_visibile, list_stemmed_visibile_actions):
    path_annotations = "/local/oignat/Action_Recog/temporal_annotation/miniclip_actions.json"
    index = 0
    dict_miniclip_actions = {}
    for miniclip in list_miniclips_visibile:
        if miniclip not in dict_miniclip_actions.keys():
            dict_miniclip_actions[miniclip] = []
        dict_miniclip_actions[miniclip].append(list_stemmed_visibile_actions[index])
        index += 1

    all_dicts_annotation = []
    all_actions = []
    for miniclip in sorted(dict_miniclip_actions.keys()):
        all_dicts_annotation.append({"miniclip": miniclip, "actions": dict_miniclip_actions[miniclip]})

    with open(path_annotations, 'w+') as fp:
        json.dump(all_dicts_annotation, fp)


def get_action_time_in_miniclip_and_in_video():
    with open("/local/oignat/Action_Recog/vlog_action_recognition/data/Video/actions_time_in_video.json", 'r') as f:
        actions_time_in_video = json.load(f)

    dict_miniclip_and_video_time = {}
    dict_miniclip_time = {}
    for miniclip in actions_time_in_video.keys():
        for [miniclip_time_in_video, action, time_action_in_video] in actions_time_in_video[miniclip]:
            time_action_in_miniclip_start = int(get_time_difference(time_action_in_video[0], miniclip_time_in_video[0]))
            time_action_in_miniclip_end = int(get_time_difference(time_action_in_video[1], miniclip_time_in_video[0]))

            # if time_action_in_miniclip_start < 10:
            #     time_action_in_miniclip_start = "0:0" + str(time_action_in_miniclip_start)
            # else:
            #     time_action_in_miniclip_start = "0:" + str(time_action_in_miniclip_start)
            #
            # if time_action_in_miniclip_end < 10:
            #     time_action_in_miniclip_end = "0:0" + str(time_action_in_miniclip_end)
            # else:
            #     time_action_in_miniclip_end = "0:" + str(time_action_in_miniclip_end)
            time_action_in_miniclip = [time_action_in_miniclip_start, time_action_in_miniclip_end]

            # TODO: maybe I will use video time for action instead of miniclip time
            if miniclip not in dict_miniclip_and_video_time.keys():
                dict_miniclip_and_video_time[miniclip] = []
                dict_miniclip_time[miniclip] = []
            dict_miniclip_and_video_time[miniclip].append(
                [miniclip_time_in_video, action.encode('utf8').lower(), time_action_in_video, time_action_in_miniclip])
            dict_miniclip_time[miniclip].append(
                [action.encode('utf8').lower(), time_action_in_miniclip])

    return dict_miniclip_time


def map_actions():
    path_miniclips = "/local/oignat/Action_Recog/vlog_action_recognition/data/miniclip_actions.json"
    with open(path_miniclips) as f:
        dict_video_actions = json.loads(f.read())

    dict_miniclip_time = get_action_time_in_miniclip_and_in_video()
    dict_map_miniclip_time = {}
    for miniclip in dict_video_actions:

        if miniclip not in dict_map_miniclip_time.keys():
            dict_map_miniclip_time[miniclip] = []

        list_actions_time = dict_miniclip_time[miniclip]
        list_actions_labels = dict_video_actions[miniclip]
        for [action_new, label] in list_actions_labels:
            ok = 0
            for [action_old, time] in list_actions_time:
                if action_old == action_new:
                    ok = 1
                    dict_map_miniclip_time[miniclip].append([action_new, time, label])
                    continue
            if ok == 0:
                for [action_old, time] in list_actions_time:
                    vector1 = text_to_vector(str(action_old))
                    vector2 = text_to_vector(str(action_new))
                    cosine = get_cosine(vector1, vector2)

                    if cosine > 0.7:
                        ok = 1
                        dict_map_miniclip_time[miniclip].append([action_new, time, label])
                        continue
            if ok == 0:
                for [action_old, time] in list_actions_time:
                    if verify_overlaps_actions(action_new, action_old):
                        ok = 1
                        dict_map_miniclip_time[miniclip].append([action_new, time, label])
                        continue

            if ok == 0:
                raise ValueError("action not found: " + action_new + " " + miniclip)

    with open("data/mapped_actions_time_label.json", 'w+') as fp:
        json.dump(dict_map_miniclip_time, fp)

    return dict_map_miniclip_time


def separate_mapped_visibile_actions(actions_time_label, video):
    visible_actions = []
    not_visible_actions = []
    list_miniclips_visibile = []
    list_miniclips_not_visibile = []
    list_time_visibile = []
    list_time_not_visibile = []

    for miniclip in actions_time_label.keys():

        if video != None:
            if video not in miniclip:
                continue

        list_actions_labels = actions_time_label[miniclip]
        for [action, time, label] in list_actions_labels:
            if label == 0:
                visible_actions.append(action)
                list_miniclips_visibile.append(miniclip)
                list_time_visibile.append(time)
            else:
                not_visible_actions.append(action)
                list_miniclips_not_visibile.append(miniclip)
                list_time_not_visibile.append(time)
    return list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions, list_time_visibile, list_time_not_visibile


def get_pos_action(action):
    text = nltk.word_tokenize(action)
    list_word_pos = nltk.pos_tag(text)
    return list_word_pos


def cosine_distance_wordembedding_method(s1, s2):
    emb_action1 = avg_GLoVe_action_emb(s1)
    emb_action2 = avg_GLoVe_action_emb(s2)
    cosine = scipy.spatial.distance.cosine(emb_action1, emb_action2)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',
          round((1 - cosine) * 100, 2), '%')


def verify_actionI3D_actionGT(clip):
    # get visible actions in that clip
    with open("data/dict_all_annotations3s.json") as f:
        dict_all_annotations10s = json.loads(f.read())

    list_visible_actions = [a for [a, label] in dict_all_annotations10s[clip + ".mp4"] if label == True]
    print(list_visible_actions)
    list_actions_clip = read_class_results(clip)
    print(list_actions_clip)

    # cosine_distance_wordembedding_method('clean up after dinner', 'washing dishes')

    # emb_action_2 = embed_fn(['washing dishes']).reshape(-1)

    tokenizer_name = 'bert-base-uncased'
    pretrained_model_name = 'bert-base-uncased'

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
    # emb_action_1 = get_bert_finetuned_embeddings(model, tokenizer, 'clean up after dinner')
    emb_action_1 = get_bert_finetuned_embeddings(model, tokenizer, 'use a little bit of mascara')
    # emb_action_5 = get_bert_finetuned_embeddings(model, tokenizer, 'clean up right after')
    emb_action_5 = get_bert_finetuned_embeddings(model, tokenizer, 'filling eyebrows')
    # emb_action_2 = get_bert_finetuned_embeddings(model, tokenizer, 'washing dishes')
    emb_action_2 = get_bert_finetuned_embeddings(model, tokenizer, 'waxing eyebrows')
    emb_action_3 = get_bert_finetuned_embeddings(model, tokenizer, 'checking tires')
    # emb_action_4 = get_bert_finetuned_embeddings(model, tokenizer, 'spray painting')
    emb_action_4 = get_bert_finetuned_embeddings(model, tokenizer, 'applying cream')

    cosine = scipy.spatial.distance.cosine(emb_action_5, emb_action_2)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',
          round((1 - cosine) * 100, 2), '%')
    cosine = scipy.spatial.distance.cosine(emb_action_5, emb_action_3)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',
          round((1 - cosine) * 100, 2), '%')
    cosine = scipy.spatial.distance.cosine(emb_action_5, emb_action_4)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',
          round((1 - cosine) * 100, 2), '%')
    cosine = scipy.spatial.distance.cosine(emb_action_1, emb_action_5)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',
          round((1 - cosine) * 100, 2), '%')


def is_action_in_clip2(action_embedding, list_action_embeddings_per_clip):
    # print(clip)
    list_sim = []
    for emb_action_1 in list_action_embeddings_per_clip:
        # print(type(action))
        # print(type(emb_action_1))
        # emb_action_1 = np.asarray(emb_action_1, dtype='float64')
        # action = np.asarray(action, dtype='float64')
        cosine = scipy.spatial.distance.cosine(emb_action_1, action_embedding)
        # sim = round((1 - cosine) * 100, 2)
        sim = 1 - cosine
        list_sim.append(sim)
        # print(action + " + " + action_clip + ": " + str(sim))
    max_sim = max(list_sim)
    threshold = 0.50
    if max_sim >= threshold:
        # print("action " + action + " is in " + clip)
        return True
    return False


#
# def is_action_in_clip(action, clip):
#     # print(clip)
#     list_actions_clip = read_class_results(clip)
#     list_sim = []
#     for action_clip in list_actions_clip:
#         sim = compute_similarity(action, action_clip)
#         list_sim.append(sim)
#         # print(action + " + " + action_clip + ": " + str(sim))
#     max_sim = max(list_sim)
#     threshold = 0.9
#     if max_sim >= threshold:
#         # print("action " + action + " is in " + clip)
#         return True
#     return False


def method_compare_actions(train_data, val_data, test_data):
    with open("data/embeddings/dict_action_embeddings_Bert2.json") as f:
        dict_action_embeddings_Bert = json.loads(f.read())

    with open("data/embeddings/dict_action_embeddings_Bert_class_I3D.json") as f:
        dict_action_embeddings_Bert_class_I3D = json.loads(f.read())

    [data_clips_train, data_actions_train, labels_train, data_actions_names_train], [data_clips_val, data_actions_val,
                                                                                     labels_val,
                                                                                     data_actions_names_val], \
    [data_clips_test, data_actions_test, labels_test, data_actions_names_test,
     data_clips_names_test] = get_features_from_data(train_data,
                                                     val_data,
                                                     test_data)

    predicted = []

    data_actions_names_test = data_actions_names_test
    data_clips_names_test = data_clips_names_test
    labels_test = labels_test

    for action, clip in tqdm(list(zip(data_actions_names_test, data_clips_names_test))):
        # result = is_action_in_clip(action, clip[:-4])
        list_actions_clip = read_class_results(clip[:-4])
        list_action_emb = []
        for action_class_I3D in list_actions_clip:
            list_action_emb.append(dict_action_embeddings_Bert_class_I3D[action_class_I3D])
        action_emb = dict_action_embeddings_Bert[action]

        result = is_action_in_clip2(action_emb, list_action_emb)
        predicted.append(result)
        # print(action + str(list_actions_clip) + str(predicted))
    # np.save("data/predicted.npy", predicted)
    # med_filt_predicted = compute_median_per_miniclip(data_actions_names_test, data_clips_names_test, predicted,
    #                                                  labels_test, med_filt_kernel_size=51)
    # predicted = med_filt_predicted
    # np.save("data/med_filt_predicted51.npy", predicted)
    print("Predicted " + str(Counter(predicted)))
    f1_test = f1_score(labels_test, predicted)
    prec_test = precision_score(labels_test, predicted)
    rec_test = recall_score(labels_test, predicted)
    acc_test = accuracy_score(labels_test, predicted)
    print("precision {0}, recall: {1}, f1: {2}".format(prec_test, rec_test, f1_test))
    print("acc_test: {:0.2f}".format(acc_test))

    list_predictions = predicted
    return predicted, list_predictions


def compute_similarity(action1, action2):
    list_word_action1 = action1.split(" ")
    list_word_action2 = action2.split(" ")

    # speed it up by collecting the synsets for all words in list_objects and list_word_action once, and taking the product of the synsets.
    allsyns1 = set(ss for word in list_word_action1 for ss in wordnet.synsets(word)[:2])
    allsyns2 = set(ss for word in list_word_action2 for ss in wordnet.synsets(word)[:2])

    if allsyns1 == set([]) or allsyns2 == set([]):
        best = 0
    else:
        best, s1, s2 = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))

    return best


def read_class_results(clip):
    path_class_results = "data/results_class_overlapping_3s/" + clip + ".txt"
    with open(path_class_results) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    list_actions = []
    for x in content:
        action = " ".join(x.split(" ")[2:])
        list_actions.append(action)
    return list_actions


def compute_action(action, use_nouns, use_particle):
    list_word_pos = get_pos_action(action)
    stemmed_word = ""
    ok_noun = 0
    if not use_nouns:
        ok_noun = 1
    for [word, pos] in list_word_pos:
        if word in ["chop", "mix"]:
            stemmed_word += word
        if 'VB' in pos:
            stemmed_word += word
            stemmed_word += " "
            if not use_nouns and not use_particle:
                break
            continue
        # if use_nouns and stemmed_word and 'NN' in pos:
        if use_nouns and 'NN' in pos:
            stemmed_word += word
            stemmed_word += " "
            ok_noun = 1
            if not use_particle:
                break
            continue
        # if use_particle and stemmed_word and ('PRT' == pos or 'RP' == pos):
        if use_particle and ('PRT' == pos or 'RP' == pos):
            stemmed_word += word
            stemmed_word += " "
            # break
            continue

    stemmed_word = stemmed_word.strip()
    if not stemmed_word:
        stemmed_word = action.split()[0]  # verb usually

    # print(action + " -> " + stemmed_word)
    return stemmed_word, ok_noun


def get_visibile_actions_verbs(visible_actions, list_miniclips_visibile,
                               use_nouns=False,
                               use_particle=False):
    # with open(path_pos_data) as f:
    #     dict_pos_actions = json.loads(f.read())

    just_visible_verbs = []
    visible_miniclips = []
    index_miniclip = 0
    for action in visible_actions:
        # if action in dict_pos_actions.keys():
        #     list_word_pos = dict_pos_actions[action]

        stemmed_word, ok_noun = compute_action(action, use_nouns, use_particle)
        # list_word_pos = get_pos_action(action)
        # stemmed_word = ""
        # ok_noun = 0
        # if not use_nouns:
        #     ok_noun = 1
        # for [word, pos] in list_word_pos:
        #     if 'VB' in pos:
        #         stemmed_word += word
        #         stemmed_word += " "
        #         if not use_nouns and not use_particle:
        #             break
        #     if use_nouns and stemmed_word and 'NN' in pos:
        #         stemmed_word += word
        #         stemmed_word += " "
        #         ok_noun = 1
        #         if not use_particle:
        #             break
        #     if use_particle and stemmed_word and ('PRT' == pos or 'RP' == pos):
        #         stemmed_word += word
        #         stemmed_word += " "
        #         break

        if stemmed_word and ok_noun:
            just_visible_verbs.append(stemmed_word.strip())
            visible_miniclips.append(list_miniclips_visibile[index_miniclip])
        index_miniclip += 1
    return just_visible_verbs, visible_miniclips


def get_most_common_visible_actions(list_stemmed_visibile_actions, k):
    counter = Counter(list_stemmed_visibile_actions)
    return (counter.most_common(k))


def get_ordered_actions_per_miniclip(visible_miniclips, list_most_common_actions):
    dict = {}
    for i in range(len(list_most_common_actions)):
        if visible_miniclips[i] not in dict.keys():
            dict[visible_miniclips[i]] = [list_most_common_actions[i]]
        else:
            dict[visible_miniclips[i]].append(list_most_common_actions[i])

    list_all_actions_ordered = []
    action_prev = ""
    miniclip = dict.keys()[0]
    video_prev = miniclip.split("mini")[0]
    for miniclip in dict.keys():
        video = miniclip.split("mini")[0]
        END_OK = 1
        if video != video_prev:
            END_OK = 0
            video_prev = video

        for action in dict[miniclip]:
            if list_most_common_actions and (
                    action in list_most_common_actions or action_prev in list_most_common_actions):
                list_all_actions_ordered.append(action)
                # END_OK = 1

            elif not list_most_common_actions:
                list_all_actions_ordered.append(action)
                # END_OK = 1
            action_prev = action
        if END_OK:
            list_all_actions_ordered.append("END")

    return list_all_actions_ordered


def create_stemmed_original_actions(path_miniclips, path_pos_data):
    with open(path_miniclips) as f:
        dict_video_actions = json.loads(f.read())

    dict_stemmed_miniclip_actions = {}
    for miniclip in dict_video_actions.keys():
        dict_stemmed_miniclip_actions[miniclip] = []
        list_actions = [action for [action, _] in dict_video_actions[miniclip]]
        list_labels = [label for [_, label] in dict_video_actions[miniclip]]
        list_stemmed_actions = stemm_list_actions(list_actions, path_pos_data)
        for i in range(len(list_stemmed_actions)):
            dict_stemmed_miniclip_actions[miniclip].append([list_stemmed_actions[i], list_labels[i]])

    with open("data/stemmed_miniclip_actions.json", 'w+') as fp:
        json.dump(dict_stemmed_miniclip_actions, fp)


def split_train_test_val_data(dict_video_actions, channel_test, channel_val):
    dict_train_data = OrderedDict()
    dict_test_data = OrderedDict()
    dict_val_data = OrderedDict()

    for channel in range(1, 11):
        if channel == channel_test or channel == channel_val:
            continue
        for key in dict_video_actions.keys():
            # if str(channel) + "p" in key or 'p' not in key[:-3]:
            if str(channel) + "p" in key:
                dict_train_data[key] = dict_video_actions[key]

    for channel in range(channel_val, channel_val + 1):
        for key in dict_video_actions.keys():
            if str(channel) + "p" in key:
                dict_val_data[key] = dict_video_actions[key]

    for channel in range(channel_test, channel_test + 1):
        for key in dict_video_actions.keys():
            if str(channel) + "p" in key:
                dict_test_data[key] = dict_video_actions[key]

    return dict_train_data, dict_test_data, dict_val_data


# lists triples of (miniclip, action, label)
def create_data(dict_train_data, dict_test_data, dict_val_data):
    train_data = []
    test_data = []
    val_data = []
    for miniclip in dict_train_data.keys():
        for [action, label] in dict_train_data[miniclip]:
            train_data.append((miniclip, action, label))

    for miniclip in dict_test_data.keys():
        for [action, label] in dict_test_data[miniclip]:
            test_data.append((miniclip, action, label))

    for miniclip in dict_val_data.keys():
        for [action, label] in dict_val_data[miniclip]:
            val_data.append((miniclip, action, label))

    return train_data, test_data, val_data


def get_data(channel_test, channel_val):
    with open("data/stemmed_miniclip_actions.json") as f:
        dict_video_actions = json.loads(f.read())

    dict_train_data, dict_test_data, dict_val_data = split_train_test_val_data(dict_video_actions, channel_test,
                                                                               channel_val)

    train_data, test_data, val_data = create_data(dict_train_data, dict_test_data, dict_val_data)
    return dict_video_actions, dict_train_data, dict_test_data, dict_val_data, train_data, test_data, val_data


def get_list_actions_for_label(dict_video_actions, miniclip, label_type):
    list_type_actions = []
    list_action_labels = dict_video_actions[miniclip]
    for [action, label] in list_action_labels:
        if label == label_type:
            list_type_actions.append(action)
    return list_type_actions


def get_nb_visible_not_visible(dict_video_actions):
    nb_visible_actions = 0
    nb_not_visible_actions = 0
    for miniclip in dict_video_actions.keys():
        nb_visible_actions += len(get_list_actions_for_label(dict_video_actions, miniclip, 0))
        nb_not_visible_actions += len(get_list_actions_for_label(dict_video_actions, miniclip, 1))
    return nb_visible_actions, nb_not_visible_actions


def print_nb_actions_miniclips_train_test_eval(dict_train_data, dict_test_data, dict_val_data):
    nb_train_actions_visible, nb_train_actions_not_visible = get_nb_visible_not_visible(dict_train_data)
    nb_train_actions = nb_train_actions_visible + nb_train_actions_not_visible

    nb_test_actions_visible, nb_test_actions_not_visible = get_nb_visible_not_visible(dict_test_data)
    nb_test_actions = nb_test_actions_visible + nb_test_actions_not_visible

    nb_val_actions_visible, nb_val_actions_not_visible = get_nb_visible_not_visible(dict_val_data)
    nb_val_actions = nb_val_actions_visible + nb_val_actions_not_visible

    print(tabulate([['nb_actions', nb_train_actions, nb_test_actions, nb_val_actions],
                    ['nb_miniclips', len(dict_train_data.keys()), len(dict_test_data.keys()),
                     len(dict_val_data.keys())]], headers=['', 'Train', 'Test', 'Eval'], tablefmt='grid'))


def process_data_channel(do_sample=True, channel_test=10, channel_val=1):
    dict_video_actions, dict_train_data, dict_test_data, dict_val_data, train_data, test_data, val_data = \
        get_data(channel_test, channel_val)

    if do_sample:
        dict_video_actions, train_data, test_data, val_data = {k: dict_video_actions[k] for k in
                                                               list(dict_video_actions.keys())[:20]}, train_data[
                                                                                                      0:200], test_data[
                                                                                                              0:20], val_data[
                                                                                                                     0:20]

        dict_val_data = {'1p0_1mini_1.mp4': dict_val_data['1p0_1mini_1.mp4']}
        train_data, test_data, val_data = create_data(dict_train_data, dict_test_data, dict_val_data)

    print_nb_actions_miniclips_train_test_eval(dict_train_data, dict_test_data, dict_val_data)

    return dict_video_actions, train_data, test_data, val_data


def get_visual_features_from_data(data_clips_train):
    # max_nb_frames = data_clips_train[0][1][0].shape[1]
    max_nb_frames = 100
    matrix_visual_features = np.zeros(
        (len(data_clips_train), max_nb_frames, 1024))

    index = 0
    for i in data_clips_train:
        padded_video_features = i[1]
        matrix_visual_features[index] = padded_video_features
        index += 1

    return matrix_visual_features


def get_features_from_data(train_data, val_data, test_data):
    [data_clips_train, data_actions_train, labels_train], [data_clips_val, data_actions_val, labels_val], \
    [data_clips_test, data_actions_test, labels_test] = train_data, val_data, test_data

    # features
    data_clips_feat_train = np.asarray([i[1] for i in data_clips_train])
    # data_clips_train = get_visual_features_from_data(data_clips_train)
    data_actions_emb_train = [i[1] for i in data_actions_train]
    data_actions_names_train = [i[0] for i in data_actions_train]
    data_clips_names_train = [i[0] for i in data_actions_train]

    data_clips_feat_val = np.asarray([i[1] for i in data_clips_val])
    # data_clips_val = get_visual_features_from_data(data_clips_val)
    data_actions_emb_val = [i[1] for i in data_actions_val]
    data_actions_names_val = [i[0] for i in data_actions_val]
    data_clips_names_val = [i[0] for i in data_actions_val]

    data_clips_feat_test = np.asarray([i[1] for i in data_clips_test])
    data_clips_names_test = [i[0] for i in data_clips_test]
    # data_clips_test = get_visual_features_from_data(data_clips_test)
    data_actions_emb_test = [i[1] for i in data_actions_test]
    data_actions_names_test = [i[0] for i in data_actions_test]

    return [data_clips_feat_train, data_actions_emb_train, labels_train, data_actions_names_train,
            data_clips_names_train], [
               data_clips_feat_val, data_actions_emb_val, labels_val, data_actions_names_val, data_clips_names_val], [
               data_clips_feat_test, data_actions_emb_test, labels_test, data_actions_names_test, data_clips_names_test]


def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]

    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.

    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return [list(a) for a in zip(*[lst[i::n] for i in range(n)])]


def process_data(train_data, test_data, val_data):
    train_labels = [label for (video, action, label) in train_data]
    test_labels = [label for (video, action, label) in test_data]
    val_labels = [label for (video, action, label) in val_data]

    train_actions = [action for (video, action, label) in train_data]
    test_actions = [action for (video, action, label) in test_data]
    val_actions = [action for (video, action, label) in val_data]

    train_video = [video for (video, action, label) in train_data]
    test_video = [video for (video, action, label) in test_data]
    val_video = [video for (video, action, label) in val_data]

    return [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], [train_video,
                                                                                                 test_video, val_video]


def split_transcript_into_sentences(path_transcripts):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    dict_transcripts_sent = {}
    with open(path_transcripts) as file:
        dict_transcripts = json.load(file)

    for channel_video in dict_transcripts.keys():
        text = " ".join(dict_["text"] for dict_ in dict_transcripts[channel_video])
        text_sentences = nlp(text)

        dict_transcripts_sent[channel_video] = [str(span) for span in text_sentences.sents]

    with open('data/transcripts_sentences.json', 'w+') as file:
        json.dump(dict_transcripts_sent, file)

    return dict_transcripts_sent


def print_results(model_name, acc_train, acc_val, acc_test, maj_val, maj_test):
    print(color.PURPLE + color.BOLD + model_name + color.END + " Train Acc {0}".format(acc_train))
    print(color.PURPLE + color.BOLD + model_name + color.END + " Val Acc {0}".format(acc_val))
    print(color.PURPLE + color.BOLD + model_name + color.END + " Test Acc {0}".format(acc_test))
    print(color.BLUE + color.BOLD + "majority_label" + color.END + " Val Acc {0}".format(maj_val))
    print(color.BLUE + color.BOLD + "majority_label" + color.END + " Test Acc {0}".format(maj_test))


def get_dict_all_annotations_visible_not():
    with open("data/stemmed_miniclip_actions.json") as file:
        stemmed_miniclip_actions = json.load(file)

    list_annotation_files = sorted(glob.glob("data/annotations/*.json"))
    dict_all_annotations_ordered = {}

    for file_path in list_annotation_files:
        with open(file_path) as f:
            video_action_dict = json.load(f)

        for video_action in video_action_dict.keys():
            video, action = video_action.split(", ")
            if video not in dict_all_annotations_ordered.keys():
                dict_all_annotations_ordered[video] = []
            if video_action_dict[video_action] != ["not visible"]:
                time = [float(i) for i in video_action_dict[video_action]]
                dict_all_annotations_ordered[video].append([action, time])
            else:
                dict_all_annotations_ordered[video].append([action, ["not visible"]])

    for video in stemmed_miniclip_actions.keys():
        list_action_label = stemmed_miniclip_actions[video]
        if video in dict_all_annotations_ordered.keys():
            for action_label in list_action_label:
                [action, label] = action_label
                if action not in [a for [a, _] in dict_all_annotations_ordered[video]]:
                    dict_all_annotations_ordered[video].append([action, ["not visible"]])

    count_n = 0
    count_v = 0
    list_videos = []
    for key in dict_all_annotations_ordered.keys():
        list_action_label = dict_all_annotations_ordered[key]
        for [action, label] in list_action_label:
            if label == ["not visible"]:
                count_n += 1
            else:
                count_v += 1
        list_videos.append(key.split("mini")[0])

    print("# videos: {0}".format(len(set(list_videos))))
    print("# miniclips: {0}".format(len(dict_all_annotations_ordered.keys())))
    print("# actions: {0}".format(count_v + count_n))
    print("# visible actions: {0}".format(count_v))
    print("# not visible actions: {0}".format(count_n))
    with open('data/dict_all_annotations_1_10channels.json', 'w+') as outfile:
        json.dump(dict_all_annotations_ordered, outfile)


def add_object_features(type):
    dict_FasterRCNN_features_clips = {}

    if type == "original":
        with open("data/embeddings/FasterRCNN/dict_FasterRCNN_features_clips.pickle", 'rb') as file:
            dict_FasterRCNN_features_clips = pickle.load(file)

    elif type == "hands":
        with open("data/embeddings/FasterRCNN/dict_FasterRCNN_hands_features_clips.pickle", 'rb') as file:
            dict_FasterRCNN_features_clips = pickle.load(file)
    else:
        print("Error argument object label type")

    dict_clip_features = {}
    for clip in dict_FasterRCNN_features_clips.keys():
        # list_labels = read_class_results(clip)
        list_features = dict_FasterRCNN_features_clips[clip]
        if not list_features:
            continue
        sum_label_embeddings = list_features[0]
        for feature in list_features[1:]:
            sum_label_embeddings += feature

        dict_clip_features[clip] = sum_label_embeddings
    return dict_clip_features


def add_object_label(type):
    dict_FasterRCNN_labels_clips = {}
    dict_action_embeddings_Bert_FasteRCNNlabels = {}

    if type == "original":
        with open("data/embeddings/FasterRCNN/dict_FasterRCNN_labels_clips.json") as file:
            dict_FasterRCNN_labels_clips = json.load(file)

        # create bert embeddings:
        path_output = "data/embeddings/FasterRCNN/dict_action_embeddings_Bert_FasteRCNN_orig.json"
        if not os.path.exists(path_output):
            set_labels = set()
            for clip in dict_FasterRCNN_labels_clips.keys():
                for s in dict_FasterRCNN_labels_clips[clip]:
                    set_labels.add(s)
            create_bert_embeddings(list(set_labels), path_output)

        with open(path_output) as file:
            dict_action_embeddings_Bert_FasteRCNNlabels = json.load(file)

    elif type == "hands":
        with open("data/embeddings/FasterRCNN/dict_FasterRCNN_hands_labels_clips.json") as file:
            dict_FasterRCNN_labels_clips = json.load(file)

        # create bert embeddings:

        path_output = "data/embeddings/FasterRCNN/dict_action_embeddings_Bert_FasteRCNN_hands.json"
        if not os.path.exists(path_output):
            set_labels = set()
            for clip in dict_FasterRCNN_labels_clips.keys():
                for s in dict_FasterRCNN_labels_clips[clip]:
                    set_labels.add(s)
            create_bert_embeddings(list(set_labels), path_output)

        with open(path_output) as file:
            dict_action_embeddings_Bert_FasteRCNNlabels = json.load(file)
    else:
        print("Error argument object label type")

    # set_labels = set()
    # for clip in dict_FasterRCNN_first3_label_str_clips.keys():
    #     for c in dict_FasterRCNN_first3_label_str_clips[clip]:
    #         set_labels.add(c)

    dict_clip_labels = {}

    # with open("data/embeddings/dict_action_embeddings_Bert_class_I3D.json") as f:
    #     dict_action_embeddings_Bert_FasteRCNNlabels_orig = json.loads(f.read())
    #

    for clip in dict_FasterRCNN_labels_clips.keys():
        # list_labels = read_class_results(clip)
        list_labels = dict_FasterRCNN_labels_clips[clip]
        if not list_labels:
            continue
        label_emb = np.array(dict_action_embeddings_Bert_FasteRCNNlabels[list_labels[0]])
        sum_label_embeddings = label_emb
        for label in list_labels[1:]:
            label_emb = np.array(dict_action_embeddings_Bert_FasteRCNNlabels[label])
            sum_label_embeddings += label_emb

        dict_clip_labels[clip] = sum_label_embeddings
    return dict_clip_labels


def add_cluster_data(dict_action_embeddings):
    with open("steve_human_action/cluster_results/dict_actions_clusters_250.json") as file:
        dict_actions_clusters = json.load(file)
    for action in dict_action_embeddings.keys():
        [cluster_nb, cluster_name, cluster_name_emb] = dict_actions_clusters[action]
        cluster_nb_normalized = (cluster_nb - 0) / (29 - 0)
        # dict_action_embeddings[action].append(cluster_nb_normalized)

        # avg action emb and cluster emb
        action_emb = dict_action_embeddings[action]
        dict_action_embeddings[action] = list(map(np.add, action_emb, cluster_name_emb))
        # dict_action_embeddings[action].append(cluster_nb_normalized)
        # dict_action_embeddings[action] = [x / 2 for x in dict_action_embeddings[action]]

        # only cluster emb
        # dict_action_embeddings[action] == cluster_name_emb

        # concatenate cluster & action emb
        # dict_action_embeddings[action] += cluster_name_emb
        return dict_action_embeddings


def get_seqs(text, tokenizer):
    max_num_words = 20000
    max_length = 22  # max nb of words in an action
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences


def get_dict_all_annotations_ordered():
    list_annotation_files = sorted(glob.glob("data/annotations/*.json"))
    dict_all_annotations_ordered = {}

    for file_path in list_annotation_files:
        with open(file_path) as f:
            video_action_dict = json.load(f)

        for video_action in video_action_dict.keys():
            video, action = video_action.split(", ")
            if video not in dict_all_annotations_ordered.keys():
                dict_all_annotations_ordered[video] = []
            if video_action_dict[video_action] != ["not visible"]:
                time = [float(i) for i in video_action_dict[video_action]]
                dict_all_annotations_ordered[video].append([action, time[0], time[1]])

    for key in dict_all_annotations_ordered.keys():
        list_elems = dict_all_annotations_ordered[key]
        list_elems.sort(key=lambda l: l[1])
        dict_all_annotations_ordered[key] = list_elems

    with open('data/dict_all_annotations_ordered.json', 'w+') as outfile:
        json.dump(dict_all_annotations_ordered, outfile)


def create_matrix(list_miniclips_visibile_new, list_stemmed_visibile_actions_new, dict_actions, key):
    list_all_actions_ordered = []
    miniclip_prev = list_miniclips_visibile_new[0]
    video_prev = miniclip_prev.split("mini")[0]
    # put 'END' only after video is ended (not miniclip)
    for (miniclip, action) in zip(list_miniclips_visibile_new, list_stemmed_visibile_actions_new):
        video = miniclip.split("mini")[0]
        # if miniclip != miniclip_prev:
        if video != video_prev:
            list_all_actions_ordered.append("END")
            # miniclip_prev = miniclip
            video_prev = video
        list_all_actions_ordered.append(action)
    # dict_actions["entire_action"] = list_all_actions_ordered
    dict_actions[key] = list_all_actions_ordered
    return dict_actions


def create_cooccurence_dictionary(list_miniclips_visibile_new, list_stemmed_visibile_actions_new):
    dict_actions = {"verb": [], "verb_particle": [], "verb_particle_nouns": [], "all_actions": []}

    dict_actions = create_matrix(list_miniclips_visibile_new, list_stemmed_visibile_actions_new, dict_actions,
                                 "all_actions")

    for (use_nouns, use_particle) in [(False, False), (False, True), (True, True)]:

        most_common_visible_actions, visible_miniclips = get_visibile_actions_verbs(list_stemmed_visibile_actions_new,
                                                                                    list_miniclips_visibile_new,
                                                                                    use_nouns,
                                                                                    use_particle)

        # list_all_actions_ordered = get_ordered_actions_per_miniclip(visible_miniclips,
        #                                                             most_common_visible_actions)  # list_stemmed_visibile_actions

        if [use_nouns, use_particle] == [False, False]:
            # dict_actions["verb"] = list_all_actions_ordered
            key = "verb"
        elif [use_nouns, use_particle] == [False, True]:
            # dict_actions["verb_particle"] = list_all_actions_ordered
            key = "verb_particle"
        elif [use_nouns, use_particle] == [True, True]:
            # dict_actions["verb_particle_nouns"] = list_all_actions_ordered
            key = "verb_particle_nouns"
        dict_actions = create_matrix(visible_miniclips, most_common_visible_actions, dict_actions, key)

    with open('steve_human_action/dict_actions_cooccurence.json', 'w+') as outfile:
        json.dump(dict_actions, outfile)

    test_cooccurence_matrix(dict_actions["verb"])
    test_cooccurence_matrix(dict_actions["verb_particle"])


# test_cooccurence_matrix(dict_actions["entire_action"])


def main():
    # for i in range(55):
    #     clip = "1p0_1mini_2" + "_{0:03}".format(i + 1)
    #     is_action_in_clip("add their toys", clip)

    clip = "1p0_1mini_1_050"
    verify_actionI3D_actionGT(clip)
    # # get_dict_all_annotations_visible_not()
    # #get_dict_all_annotations_ordered()
    #
    # with open("data/dict_all_annotations_ordered.json") as file:
    #     dict_all_annotations_ordered = json.load(file)
    #
    # list_stemmed_visibile_actions_new = []
    # list_miniclips_visibile_new = []
    #
    # for miniclip in dict_all_annotations_ordered.keys():
    #     list_actions_labels = dict_all_annotations_ordered[miniclip]
    #     for [action, time_s, time_e] in list_actions_labels:
    #         list_stemmed_visibile_actions_new.append(action)
    #         list_miniclips_visibile_new.append(miniclip)
    #
    # with open("data/embeddings/dict_action_embeddings_ELMo.json") as file:
    #     dict_action_embeddings_ELMo = json.load(file)
    #
    # print(len(dict_action_embeddings_ELMo.keys()))
    # print(len(list(set(list_stemmed_visibile_actions_new))))
    # create_cooccurence_dictionary(list_miniclips_visibile_new, list_stemmed_visibile_actions_new)

    # path_miniclips = "data/miniclip_actions.json"
    # path_pos_data = "data/dict_action_pos_concreteness.json"
    # path_list_actions = "data/stats/list_actions.csv"
    # path_transcripts = "/local/oignat/Action_Recog/vlog_action_recognition/data/Video/new_videos_captions/new_transcripts.json"
    # split_transcript_into_sentences(path_transcripts)

    # create_stemmed_original_actions(path_miniclips, path_pos_data)
    # list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions = separate_visibile_actions(
    #     path_miniclips, video=None)
    # all_actions = visible_actions + not_visible_actions

    # print("---- Looking at visible actions ----")
    # stats(all_actions, path_pos_data)
    #
    # list_stemmed_not_visibile_actions = stemm_list_actions(not_visible_actions, path_pos_data)
    # list_stemmed_visibile_actions = stemm_list_actions(visible_actions, path_pos_data)

    # with open('data/list_stemmed_visibile_actions.txt', 'w') as f:
    #     for item in list_stemmed_visibile_actions:
    #         f.write("%s\n" % item)

    # write_list_to_csv(list_miniclips_visibile, list(visible_actions), list_stemmed_visibile_actions,
    #                   list_miniclips_not_visibile, list(not_visible_actions), list_stemmed_not_visibile_actions)

    # with open("data/stemmed_miniclip_actions.json") as file:
    #     stemmed_miniclip_actions = json.load(file)
    #
    # list_stemmed_visibile_actions = []
    # not_visible_actions = []
    # list_miniclips_visibile = []
    # list_miniclips_not_visibile = []
    #
    # for miniclip in stemmed_miniclip_actions.keys():
    #
    #     list_actions_labels = stemmed_miniclip_actions[miniclip]
    #     for [action, label] in list_actions_labels:
    #         if label == 0:
    #             list_stemmed_visibile_actions.append(action)
    #             list_miniclips_visibile.append(miniclip)
    #         else:
    #             not_visible_actions.append(action)
    #             list_miniclips_not_visibile.append(miniclip)
    #
    # with open("data/dict_action_embeddings_ELMo.json") as file:
    #     dict_action_embeddings_ELMo = json.load(file)
    #
    # visible_actions_new = []
    # list_miniclips_visibile_new = []
    # list_stemmed_visibile_actions_new = []
    # for action_i in range(len(list_stemmed_visibile_actions)):
    #     if list_stemmed_visibile_actions[action_i] in dict_action_embeddings_ELMo.keys():
    #         list_miniclips_visibile_new.append(list_miniclips_visibile[action_i])
    #         list_stemmed_visibile_actions_new.append(list_stemmed_visibile_actions[action_i])
    #
    # print(len(list(set(list_stemmed_visibile_actions_new))))
    # print(len(dict_action_embeddings_ELMo.keys()))
    #
    #
    # just_visible_verbs, visible_miniclips = get_visibile_actions_verbs(list_stemmed_visibile_actions_new,
    #                                                                    list_miniclips_visibile_new, use_nouns=False,
    #                                                                    use_particle=False)
    # most_common_visible_verbs = [action for action, count in
    #                              get_most_common_visible_actions(just_visible_verbs,
    #                                                              20)]  # list_stemmed_visibile_actions

    # list_verbs = ["add", "use", "put", "make", "do", "take", "get", "go", "clean", "give"]
    # analyze_verbs(list_verbs, list_stemmed_visibile_actions)

    # time_flow_actions(path_list_actions, visible=True)
    # time_flow_actions(path_list_actions, visible=False)
    #
    #
    # list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions = separate_visibile_actions(
    #     path_miniclips, video="1p0")
    #
    # list_stemmed_not_visibile_actions = stemm_list_actions(not_visible_actions, path_pos_data)
    # list_stemmed_visibile_actions = stemm_list_actions(visible_actions, path_pos_data)
    #
    # write_mini_action_time_to_csv(list_miniclips_visibile, list_stemmed_visibile_actions,
    #                               list_miniclips_not_visibile, list_stemmed_not_visibile_actions)

    # write_json_for_annotations(list_miniclips_visibile, list_stemmed_visibile_actions)


# test_coocurence_matrix(list_stemmed_visibile_actions)

if __name__ == '__main__':
    # map_actions()
    # get_extra_data_coin()
    main()

    # path_root = "/local/oignat/Action_Recog/large_data/"
    #
    # path_old_urls = path_root + "all_transcripts/video_urls/"
    # path_new_urls = path_root + "youtube_data/video_urls/"
    # map_old_to_new_urls(path_old_urls, path_new_urls)
