import csv
import glob
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from compute_text_embeddings import create_bert_embeddings


def collect_data(path_annotations):
    file_to_open = path_annotations / "annotations4p0.json"

    with open(file_to_open) as f:
        data = json.load(f)

    new_dict = {}
    for key in data.keys():
        miniclip, action = key.split(", ")
        time_info = data[key]
        if time_info == ['not visible']:
            continue
        if miniclip not in new_dict.keys():
            new_dict[miniclip] = []

        new_dict[miniclip].append([action, time_info])

    return new_dict


def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def find_overlapping_actions(new_dict):
    dict_overlap = {}
    for miniclip in new_dict.keys():
        list_action_time = new_dict[miniclip]
        list_time = []
        for [action, time] in list_action_time:
            s = float(time[0])
            e = float(time[1])
            list_time.append([s, e])
        if len(list_time) > 1:
            for i in range(0, len(list_time) - 1):

                if str(list_action_time[i]) not in dict_overlap.keys():
                    dict_overlap[miniclip + " " + str(list_action_time[i])] = []

                set_overlap_actions = set()
                set_perfect_actions = set()
                for j in range(i + 1, len(list_time)):
                    if (getOverlap(list_time[i], list_time[j])) \
                            >= (list_time[j][1] - list_time[j][0]):
                        set_perfect_actions.add(str(list_action_time[j]))
                    elif (getOverlap(list_time[i], list_time[j])) > 0:

                        set_overlap_actions.add(str(list_action_time[j]))

                dict_overlap[miniclip + " " + str(list_action_time[i])].append(set_overlap_actions)
                dict_overlap[miniclip + " " + str(list_action_time[i])].append(set_perfect_actions)

    for action_time in dict_overlap.keys():
        if dict_overlap[action_time] != [set(), set()]:
            print("key: " + action_time)
            print("perfect overlap: " + str(list(dict_overlap[action_time][1])))
            print("some overlap: " + str(list(dict_overlap[action_time][0])))
            print("----------------------")


def plot_hist_length_actions(annotations, channel=None):
    list_duration = []
    max_nb_words = 0
    for miniclip in annotations.keys():
        if channel:
            if channel not in miniclip:
                continue
        list_action_label = annotations[miniclip]
        for [action, label] in list_action_label:
            if label != ["not visible"]:
                [time_s, time_e] = label
                if time_e == time_s == -1:
                    rounded_duration = "-1"
                else:
                    action_duration = int(time_e - time_s)
                    # 5 -> 0; 6 -> 10
                    rounded_duration = str(int(round(action_duration, -1)))
                if int(rounded_duration) < 0:
                    print(miniclip, action)
                if len(action.split()) > max_nb_words:
                    max_nb_words = len(action.split())

                list_duration.append(rounded_duration)

    counter = Counter(list_duration)
    # counter = counter.most_common()
    counter = sorted(counter.items())
    labels, values = zip(*counter)
    # for l in ['-1', '0', '10', '20', '30', '40', '50', '60']:
    # for l in ['0', '10', '20', '30', '40', '50', '60']:
    for l in ['0', '10', '20', '30', '40', '50', '60']:
        if l not in labels:
            counter.append((l, 0))

    # counter = sorted(counter.items())
    print(counter)
    labels, values = zip(*counter)
    indexes = np.arange(len(labels))
    width = 1

    # # plt.bar(indexes, values, width, color=['yellow', 'red', 'green', 'blue', 'cyan', "pink", "orange", "purple"])
    # plt.bar(indexes, values, width)
    # plt.xticks(indexes + width * 0.5, labels)
    # plt.ylabel('count', fontsize=18)
    # plt.xlabel('rounded seconds', fontsize=16)
    # plt.show()
    # # ax = sns.countplot(x="rounded seconds",data=counter)
    #
    # print("max # words action: " + str(max_nb_words))


def count_how_many_times_actions_overlap():
    with open("data/dict_all_annotations_1_10channels.json") as file:
        annotations = json.load(file)

    count_overlap = 0
    count_inclusion = 0
    count_exact_time = 0
    for miniclip in annotations.keys():
        list_intervals = []
        list_actions = []
        list_action_label = annotations[miniclip]
        for [action, label] in list_action_label:
            if label != ["not visible"]:
                [time_s, time_e] = label
                interval = range(int(time_s), int(time_e))
                list_intervals.append(interval)
                list_actions.append(action)
        for x in range(0, len(list_intervals) - 1):
            for y in range(x + 1, len(list_intervals)):
                if set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[x]) or set(
                        list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[y]):
                    count_inclusion += 1
                    # if set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[x]):
                    # print(miniclip, list_actions[x] + " -> " + list_actions[y])
                    # print(list_intervals[x], list_intervals[y])
                    # elif set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[y]):
                    # print(miniclip, list_actions[y] + " -> " + list_actions[x])
                    # print(list_intervals[y], list_intervals[x])
                    if set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[x]) and set(
                            list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[y]):
                        # print(miniclip, list_actions[y] + " = " + list_actions[x])
                        # print(list_intervals[y], list_intervals[x])
                        count_exact_time += 1
                elif set(list_intervals[x]).intersection(list_intervals[y]):
                    count_overlap += 1
                    print(miniclip, list_actions[y] + " & " + list_actions[x])

    print(count_overlap)
    print(count_inclusion)
    print(count_exact_time)


def change_format(initial):
    new_format_dict = {}
    with open(initial) as file:
        predicted = json.load(file)

    for video_action in predicted.keys():
        video, action = video_action.split(", ")
        if video not in new_format_dict.keys():
            new_format_dict[video] = []
            time = [float(i) for i in predicted[video_action][0]]
            new_format_dict[video].append([action, time])

    return new_format_dict


def read_COIN():
    with open("data/RelatedWorkDatasets/COIN.json") as file:
        coin_data = json.load(file)

    data = coin_data["database"]
    list_duration = []
    list_all_actions = set()
    for key in data.keys():
        content = data[key]
        for i in range(len(content["annotation"])):
            segment_time = content["annotation"][i]["segment"]
            action = content["annotation"][i]["label"]
            action_duration = int(segment_time[1] - segment_time[0])

            # 5 -> 0; 6 -> 10
            rounded_duration = str(int(round(action_duration, -1)))
            list_duration.append(rounded_duration)
            list_all_actions.add(action)

    # create_bert_embeddings(list_all_actions)
    counter = Counter(list_duration)
    # counter = counter.most_common()
    counter = sorted(counter.items())
    print(counter)
    sum1 = 0
    for c, v in counter[:2]:
        sum1 += v
    print("1-15: " + str(sum1))

    sum2 = 0
    for c, v in counter[2:]:
        sum2 += v
    print("16-175: " + str(sum2))

    nb_total_actions = sum1 + sum2
    print("COIN:")
    print("nb_total_actions: " + str(nb_total_actions))
    print("nb_0-15s_actions relative to total: " + str(sum1 / nb_total_actions * 100))
    print("nb_16-60s_actions relative to total: " + str(sum2 / nb_total_actions * 100))


def read_howto100m():
    with open("data/RelatedWorkDatasets/HowTo100M/caption.json") as file:
        data = json.load(file)


    list_duration = []
    # list_all_actions = set()
    for key in tqdm(data.keys()):
        data_start_end = zip(data[key]["start"], data[key]["end"])

        for [start, end] in data_start_end:
            action_duration = int(end - start)

            # 5 -> 0; 6 -> 10
            rounded_duration = str(int(round(action_duration, -1)))
            list_duration.append(rounded_duration)
            # list_all_actions.add(action)

    # create_bert_embeddings(list_all_actions)
    counter = Counter(list_duration)
    # counter = counter.most_common()
    counter = sorted(counter.items())
    print(counter)
    sum1 = 0
    for c, v in counter[:2]:
        sum1 += v
    print("1-15: " + str(sum1))

    sum2 = 0
    for c, v in counter[2:]:
        sum2 += v
    print("16-175: " + str(sum2))

    nb_total_actions = sum1 + sum2
    print("COIN:")
    print("nb_total_actions: " + str(nb_total_actions))
    print("nb_0-15s_actions relative to total: " + str(sum1 / nb_total_actions * 100))
    print("nb_16-60s_actions relative to total: " + str(sum2 / nb_total_actions * 100))


def read_CrossTask():
    list_csv_files = sorted(glob.glob("data/RelatedWorkDatasets/crosstask/annotations/*.csv"))
    list_duration = []
    for file_path in list_csv_files:
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for data in csv_reader:
                action_duration = int(float(data[2]) - float(data[1]))
                rounded_duration = str(int(round(action_duration, -1)))
                list_duration.append(rounded_duration)

    counter = Counter(list_duration)
    # counter = counter.most_common()
    counter = sorted(counter.items())
    print(counter)
    sum1 = 0
    for c, v in counter[:2]:
        sum1 += v
    print("1-15: " + str(sum1))

    sum2 = 0
    for c, v in counter[2:]:
        sum2 += v
    print("16-175: " + str(sum2))
    nb_total_actions = sum1 + sum2
    print("CrossTask:")
    print("nb_total_actions: " + str(nb_total_actions))
    print("nb_0-15s_actions relative to total: " + str(sum1 / nb_total_actions * 100))
    print("nb_16-60s_actions relative to total: " + str(sum2 / nb_total_actions * 100))


def plot_nb_actions_per_channel():
    sns.set(style="whitegrid")
    sns.set(rc={"font.style": "normal",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "axes.labelcolor": "black",
                'axes.labelsize': 30,
                'figure.figsize': (20.0, 10.0),
                'xtick.labelsize': 35,
                'font.size': 35,
                'ytick.labelsize': 35})

    # Load the example Titanic dataset
    # titanic = sns.load_dataset("titanic")
    # print(titanic[:5])
    #
    # # Draw a nested barplot to show survival for class and sex
    # g = sns.catplot(x="class", y="survived", hue="sex", data=titanic,
    #                 height=6, kind="bar", palette="muted")
    # g.despine(left=True)
    # g.set_ylabels("survival probability")
    # plt.show()

    # list of name, degree, score
    nb_actions = [1136, 1200, 475, 157, 72, 69, 30]
    time_span = ["0-5s", "6-15s", "16-25s", "26-35s", "36-45s", "46-55s", "56-60s"]
    nb_total_actions = sum(nb_actions)
    print("Mine:")
    print("nb_total_actions: " + str(nb_total_actions))
    print("nb_0-15s_actions relative to total: " + str(sum(nb_actions[0:2]) / nb_total_actions * 100))
    print("nb_16-60s_actions relative to total: " + str(sum(nb_actions[2:]) / nb_total_actions * 100))

    # list of name, degree, score
    nb_actions = [1548, 10328, 500, 21, 3, 0, 2, 0, 3]
    time_span = ["0-5s", "6-15s", "16-25s", "26-35s", "36-45s", "46-55s", "56-60s"]
    nb_total_actions = sum(nb_actions)
    print("Charades:")
    print("nb_total_actions: " + str(nb_total_actions))
    print("nb_0-15s_actions relative to total: " + str(sum(nb_actions[0:2]) / nb_total_actions * 100))
    print("nb_16-60s_actions relative to total: " + str(sum(nb_actions[2:]) / nb_total_actions * 100))

    # # dictionary of lists
    # dict = {'time span': time_span, '#actions': nb_actions}
    #
    # df = pd.DataFrame(dict)
    #
    # # saving the dataframe
    # df.to_csv('data/data_to_plot/action_duration.csv')

    # https://python-graph-gallery.com/100-calling-a-color-with-seaborn/

    tips = pd.read_csv("data/data_to_plot/action_duration.csv")

    ax = sns.barplot(x="time span", y="#actions", data=tips, color="royalblue")
    ax.set_title('Action length distribution', fontsize=45)
    plt.show()


def main():
    # plot_nb_actions_per_channel()
    # read_COIN()
    read_howto100m()
    # read_CrossTask()
    # count_how_many_times_actions_overlap()
    # new_format_dict = change_format("data/results/dict_predicted_MPU + ELMo + 651p0.json")
    # with open("data/dict_all_annotations_1_10channels.json") as file:
    #     annotations = json.load(file)
    # plot_hist_length_actions(annotations)
    # plot_hist_length_actions(annotations, "4p0")
    # count_how_many_times_actions_overlap()
    # path_annotations = Path("data/annotations/")
    #
    # new_dict = collect_data(path_annotations)
    # find_overlapping_actions(new_dict)


if __name__ == "__main__":
    main()
