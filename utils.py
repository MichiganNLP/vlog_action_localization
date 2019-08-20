from __future__ import print_function, absolute_import, unicode_literals, division
import json
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer

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
    #measure_nb_words(list_actions)
    measure_verb_distribution(list_actions, path_pos_data)


def write_list_to_csv(list_actions, list_stemmed_actions):
    list_all = []
    list_all.append(list_actions)
    list_all.append(list_stemmed_actions)

    df = pd.DataFrame(list_all)
    df = df.transpose()
    df.columns = ["Visible Actions", "Stemmed Visible Actions"]
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

    print("---- Looking at visible actions ----")
    stats(all_actions, path_pos_data)

    list_stemmed_actions = stemm_list_actions(visible_actions, path_pos_data)
    # write_list_to_csv(list(visible_actions), list_stemmed_actions)

    list_verbs = ["add", "use", "put", "make", "do", "take", "get", "go", "clean", "give"]
    analyze_verbs(list_verbs, list_stemmed_actions)


if __name__ == '__main__':
    main()
