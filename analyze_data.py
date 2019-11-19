import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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


def plot_hist_length_actions():
    with open("data/dict_all_annotations_1_7channels.json") as file:
        annotations = json.load(file)

    list_duration = []
    for miniclip in annotations.keys():
        list_action_label = annotations[miniclip]
        for [action, label] in list_action_label:
            if label != ["not visible"]:
                [time_s, time_e] = label
                action_duration = int(time_e - time_s)
                # 5 -> 0; 6 -> 10
                rounded_duration = str(int(round(action_duration, -1)))
                list_duration.append(rounded_duration)

    counter = Counter(list_duration)
    counter = counter.most_common()
    labels, values = zip(*counter)

    indexes = np.arange(len(labels))
    width = 1

    print(counter)

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.ylabel('count', fontsize=18)
    plt.xlabel('rounded seconds', fontsize=16)
    plt.show()
    # ax = sns.countplot(x="rounded seconds",data=counter)


def count_how_many_times_actions_overlap():
    with open("data/dict_all_annotations_1_7channels.json") as file:
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
                if set(list_intervals[x]).intersection(list_intervals[y]):
                    count_overlap += 1
                if set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[x]) or set(
                        list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[y]):
                    count_inclusion += 1
                    #if set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[x]):
                        # print(miniclip, list_actions[x] + " -> " + list_actions[y])
                        # print(list_intervals[x], list_intervals[y])
                    #elif set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[y]):
                        # print(miniclip, list_actions[y] + " -> " + list_actions[x])
                        # print(list_intervals[y], list_intervals[x])
                    if set(list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[x]) and set(
                            list_intervals[x]).intersection(list_intervals[y]) == set(list_intervals[y]):
                        print(miniclip, list_actions[y] + " = " + list_actions[x])
                        print(list_intervals[y], list_intervals[x])
                        count_exact_time += 1


    print(count_overlap)
    print(count_inclusion)
    print(count_exact_time)


def main():
    # plot_hist_length_actions()
    count_how_many_times_actions_overlap()
    # path_annotations = Path("data/annotations/")
    #
    # new_dict = collect_data(path_annotations)
    # find_overlapping_actions(new_dict)


if __name__ == "__main__":
    main()
