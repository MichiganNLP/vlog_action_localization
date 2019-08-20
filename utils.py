from __future__ import print_function, absolute_import, unicode_literals, division
import json
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer

plt.style.use('ggplot')


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
    plt.title("Number of words per visible action")
    # plt.show()
    # plt.savefig('data/stats/nb_words_per_all_action.png')


def stem_word(word):
    handcraft_rules = {'ad': 'add', 'tri': 'try', 'saut': 'saute', 'danc': 'dance', 'remov': 'remove', 'has': 'have',
                       'did': 'do', 'made': 'make', 'cleans': 'clean', 'creat': 'create', 'dri': 'dry', 'done': 'do',
                       'ran': 'run','ate':'eat','cleaning':'clean',
                       'snuggl': 'snuggle', 'subscrib': 'subscribe', 'squeez': 'squeeze', 'chose': 'choose',
                       'bundl': 'bundle', 'decid': 'decide', 'empti': 'empty', 'wore': 'wear', 'starv': 'starve',
                       'increas': 'increase', 'incorpor':'incorporate',
                       'purchas': 'purchase', 'laid': 'lay', 'rins': 'rinse', 'saw': 'see',
                       'goe': 'go', 'appli': 'apply', 'diffus': 'diffuse',
                       'combin': 'combine', 'shown': 'show', 'stapl': 'staple', 'burnt': 'burn', 'imagin': 'imagine',
                       'achiev': 'achieve','sped':'speed','chose':'choose', 'carri':'carry'}
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
    plt.title("First 10 diff verbs for all actions actions")
    #plt.show()
    plt.savefig('data/stats/first_10_verbs_all_actions.png')


def stats(list_actions, path_pos_data):
    # measure_nb_words(list_actions)
    measure_verb_distribution(list_actions, path_pos_data)


def write_list_to_csv(list_actions, list_stemmed_actions):
    list_all = []
    list_all.append(list_actions)
    list_all.append(list_stemmed_actions)

    df = pd.DataFrame(list_all)
    df = df.transpose()
    df.columns=["Visible Actions", "Stemmed Visible Actions"]
    df.to_csv('data/list_actions.csv', index=False)


def separate_visibile_actions(path_miniclips):
    visible_actions = []
    not_visible_actions = []

    with open(path_miniclips) as f:
        dict_video_actions = json.loads(f.read())

    for miniclip in dict_video_actions.keys():
        list_actions_labels = dict_video_actions[miniclip]
        for [action, label] in list_actions_labels:
            if label == 0:
                visible_actions.append(action)
            else:
                not_visible_actions.append(action)
    return visible_actions, not_visible_actions


def main():
    path_miniclips = "/local/oignat/Action_Recog/vlog_action_recognition/data/miniclip_actions.json"
    path_pos_data = "/local/oignat/Action_Recog/vlog_action_recognition/data/dict_action_pos_concreteness.json"
    visible_actions, not_visible_actions = separate_visibile_actions(path_miniclips)
    all_actions = visible_actions + not_visible_actions
    stats(all_actions, path_pos_data)

    #list_stemmed_actions = stemm_list_actions(visible_actions, path_pos_data)
    #write_list_to_csv(list(visible_actions), list_stemmed_actions)


if __name__ == '__main__':
    main()
