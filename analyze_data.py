import json
from pathlib import Path


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


def main():
    path_annotations = Path("data/annotations/")

    new_dict = collect_data(path_annotations)
    find_overlapping_actions(new_dict)


if __name__ == "__main__":
    main()
