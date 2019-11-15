import json
from utils_data_text import stemm_list_actions


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



def main():
    path_results = "data/annotations/annotations1p0.json"
    path_pos_data = "/local/oignat/Action_Recog/vlog_action_recognition/data/dict_action_pos_concreteness.json"
    # compare_data()
    test_alignment(path_pos_data)


if __name__ == '__main__':
    main()
