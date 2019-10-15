import json
from video_processing.experiments import separate_mapped_visibile_actions
from video_processing.utils_data import stemm_list_actions


def compare_data():
    with open("data/stemmed_actions_miniclip_time1p0.json") as f:
        proposed_1p0 = json.loads(f.read())

    with open("data/annotations/annotations1p0.json") as f:
        groundtruth_1p0 = json.loads(f.read())

    if len(groundtruth_1p0.keys()) != len(proposed_1p0.keys()):
        raise ValueError("GT data and proposed method don't have the same nb of elements!")

    for miniclip_action in groundtruth_1p0.keys():
        miniclip, action = miniclip_action.split(", ")
        if miniclip_action not in proposed_1p0.keys():
            raise ValueError(miniclip_action + " in GT data but not in proposed method!")


def test_alignment(path_pos_data):
    with open("data/mapped_actions_time_label.json") as f:
        actions_time_label = json.loads(f.read())

    # extract the visible ones and stem them
    list_miniclips_visibile, visible_actions, list_miniclips_not_visibile, not_visible_actions, list_time_visibile, list_time_not_visibile = separate_mapped_visibile_actions(
        actions_time_label, "1p0")
    list_stemmed_visibile_actions = stemm_list_actions(visible_actions, path_pos_data)

    miniclips_list_stemmed_visibile_actions = {}
    for index in range(len(list_stemmed_visibile_actions)):
        miniclip = list_miniclips_visibile[index]
        action = list_stemmed_visibile_actions[index]
        time = list_time_visibile[index]
        miniclip_action = miniclip + ", " + action
        miniclips_list_stemmed_visibile_actions[miniclip_action] = time

    with open("data/stemmed_actions_miniclip_time1p0.json", 'w+') as fp:
        json.dump(miniclips_list_stemmed_visibile_actions, fp)


def main():
    path_results = "data/annotations/annotations1p0.json"
    path_pos_data = "/local/oignat/Action_Recog/vlog_action_recognition/data/dict_action_pos_concreteness.json"
    compare_data()


if __name__ == '__main__':
    main()
