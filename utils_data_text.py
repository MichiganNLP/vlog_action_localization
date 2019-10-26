from __future__ import print_function, absolute_import, unicode_literals, division

import itertools

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
from datetime import datetime
import re, math
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
from nltk import word_tokenize
from tabulate import tabulate
import tensorflow as tf
import tensorflow_hub as hub

WORD = re.compile(r'\w+')

plt.style.use('ggplot')

embeddings_index = dict()
with open("/local/oignat/Action_Recog/vlog_action_recognition/data/glove.6B.50d.txt") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

dimension_embedding = len(embeddings_index.get("example"))


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
                       'ran': 'run', 'ate': 'eat', 'cleaning': 'clean',
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


def stemm_list_actions(list_actions, path_pos_data):
    stemmed_actions = []
    with open(path_pos_data) as f:
        dict_pos_actions = json.loads(f.read())

    for action in list_actions:
        if action in dict_pos_actions.keys():
            list_word_pos = dict_pos_actions[action]
            for [word, pos, concr_score] in list_word_pos:
                if 'VB' in pos:
                    stemmed_word = stem_word(word)
                    words = action.split()
                    replaced = " ".join([stemmed_word if wd == word else wd for wd in words])
                    action = replaced

            stemmed_actions.append(action)

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
    print(matrix)
    sorted_indices_row = np.argsort(matrix.sum(axis=1))[::-1]  # sort indices by row (sum of the elemnts) & reverse list (to be descending order)
    sorted_matrix = matrix[:, sorted_indices_row]


    sorted_indices_col = np.argsort(sorted_matrix.sum(axis=0))[
                     ::-1]  # sort indices by column (sum of the elemnts) & reverse list (to be descending order)
    sorted_matrix = sorted_matrix[:, sorted_indices_col]

    print(sorted_matrix)

    print("row")
    print(sorted_indices_row)

    print("col")
    print(sorted_indices_col)

    sorted_indices_row_col = [0] * len(sorted_indices_row)
    for i in range(len(sorted_indices_row)):
        pos = sorted_indices_col[i]
        sorted_indices_row_col[i] = sorted_indices_row[pos]

    print("row col")
    print(sorted_indices_row_col)
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


    sorted_vocab_index_v = [v for (v,_) in sorted(sorted_vocab_index.items(), key=lambda kv: kv[1])]

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

    # column_values = list(data_matrix_sorted.columns.values)
    # print(column_values[0:5])
    # maxValuesObj = data_matrix_sorted.max()
    # print(maxValuesObj)
    # first_data_matrix = data_matrix_sorted.nlargest(30, column_values[0]) # sort the rows
    # first_data_matrix = first_data_matrix.iloc[:,:30] # first 50 columns

    first_data_matrix = data_matrix_sorted.iloc[:, :50]  # first 50 columns
    first_data_matrix = first_data_matrix.iloc[:50, :]  # first 50 rows



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

    #sns.clustermap(data_matrix, xticklabels=1, yticklabels=1,cmap=cmap)




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


def get_visibile_actions_verbs(path_pos_data, list_visibile_actions, use_nouns = False, use_particle = False):
    with open(path_pos_data) as f:
        dict_pos_actions = json.loads(f.read())

    just_visible_verbs = []
    for action in list_visibile_actions:
        if action in dict_pos_actions.keys():
            list_word_pos = dict_pos_actions[action]
            stemmed_word = ""
            ok_noun = 0
            if not use_nouns:
                ok_noun = 1
            for [word, pos, concr_score] in list_word_pos:
                if 'VB' in pos:
                    stemmed_word += stem_word(word)
                    stemmed_word += " "
                    if not use_nouns and not use_particle:
                        break
                if use_nouns and stemmed_word and 'NN' in pos:
                    stemmed_word += word
                    stemmed_word += " "
                    ok_noun = 1
                    if not use_particle:
                        break
                if use_particle and stemmed_word and ('PRT' == pos or 'RP' == pos):
                    stemmed_word += word
                    stemmed_word += " "
                    break

            if stemmed_word and ok_noun:
                just_visible_verbs.append(stemmed_word.strip())
    return just_visible_verbs



def get_most_common_visible_actions(list_stemmed_visibile_actions, k):

    counter = Counter(list_stemmed_visibile_actions)
    return (counter.most_common(k))


def get_ordered_actions_per_miniclip(list_stemmed_visibile_actions, list_miniclips_visibile,
                                     list_most_common_actions=[]):
    dict = {}
    for i in range(len(list_stemmed_visibile_actions)):
        if list_miniclips_visibile[i] not in dict.keys():
            dict[list_miniclips_visibile[i]] = [list_stemmed_visibile_actions[i]]
        else:
            dict[list_miniclips_visibile[i]].append(list_stemmed_visibile_actions[i])

    list_all_actions_ordered = []
    action_prev = ""
    for miniclip in dict.keys():
        END_OK = 0
        for action in dict[miniclip]:
            if list_most_common_actions and (
                    action in list_most_common_actions or action_prev in list_most_common_actions):
                list_all_actions_ordered.append(action)
                END_OK = 1

            elif not list_most_common_actions:
                list_all_actions_ordered.append(action)
                END_OK = 1
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



def embed_elmo2():
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


def create_action_emb(action, type):
    if type == "GloVe":
        return avg_GLoVe_action_emb(action)
    elif type == "ELMo":
        embed_fn = embed_elmo2()
        return embed_fn([action]).reshape(1024)
    else:
        raise ValueError("Wrong action emb type")

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


def create_average_action_embedding(list_actions):

    embedding_matrix_actions = np.zeros((len(list_actions), dimension_embedding))
    index = 0
    for action in list_actions:
        average_word_embedding = avg_GLoVe_action_emb(action)
        embedding_matrix_actions[index] = average_word_embedding
        index += 1
    return embedding_matrix_actions


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


def main():
    path_miniclips = "data/miniclip_actions.json"
    path_pos_data = "data/dict_action_pos_concreteness.json"
    path_list_actions = "data/stats/list_actions.csv"
    path_transcripts = "/local/oignat/Action_Recog/vlog_action_recognition/data/Video/new_videos_captions/new_transcripts.json"
    split_transcript_into_sentences(path_transcripts)

    #create_stemmed_original_actions(path_miniclips, path_pos_data)
    # list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions = separate_visibile_actions(
    #     path_miniclips, video=None)
    # all_actions = visible_actions + not_visible_actions

    #print("---- Looking at visible actions ----")
    # stats(all_actions, path_pos_data)
    #
    # list_stemmed_not_visibile_actions = stemm_list_actions(not_visible_actions, path_pos_data)
    # list_stemmed_visibile_actions = stemm_list_actions(visible_actions, path_pos_data)

    # with open('data/list_stemmed_visibile_actions.txt', 'w') as f:
    #     for item in list_stemmed_visibile_actions:
    #         f.write("%s\n" % item)

    # write_list_to_csv(list_miniclips_visibile, list(visible_actions), list_stemmed_visibile_actions,
    #                   list_miniclips_not_visibile, list(not_visible_actions), list_stemmed_not_visibile_actions)

    # just_visible_verbs = get_visibile_actions_verbs(path_pos_data, list_stemmed_visibile_actions, use_nouns = True, use_particle=True)
    #
    # list_most_common_actions = [action for action, count in
    #                             get_most_common_visible_actions(just_visible_verbs, 10)] #list_stemmed_visibile_actions
    # print(list_most_common_actions)
    # # list_most_common_actions = [l for l in list_most_common_actions if l not in ['make it', 'show', 'do this', 'make sure']]
    # #
    # # print(list_most_common_actions)
    # list_all_actions_ordered = get_ordered_actions_per_miniclip(just_visible_verbs, list_miniclips_visibile,
    #                                                             list_most_common_actions) #list_stemmed_visibile_actions
    #
    #
    # print(len(list_all_actions_ordered))
    # test_cooccurence_matrix(list_all_actions_ordered)

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

    main()

    # path_root = "/local/oignat/Action_Recog/large_data/"
    #
    # path_old_urls = path_root + "all_transcripts/video_urls/"
    # path_new_urls = path_root + "youtube_data/video_urls/"
    # map_old_to_new_urls(path_old_urls, path_new_urls)
