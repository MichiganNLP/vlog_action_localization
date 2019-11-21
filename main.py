import os
import json
from utils_data_video import get_clip_time_per_miniclip, create_action_clip_labels, load_data_from_I3D, create_clips

channels = ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1", "6p0", "6p1", "7p0", "7p1"]


# channels = ["6p0", "6p1", "7p0", "7p1"]

def compare_json_files():
    with open("data/dict_action_embeddings_ELMo.json") as file:
        dict_action_embeddings_ELMo = json.load(file)

    with open("data/dict_actions_cooccurence.json") as file:
        dict_actions_cooccurence = json.load(file)

    print(len(dict_action_embeddings_ELMo.keys()))
    list_coocurence = list(set(dict_actions_cooccurence["entire_action"]))

    print(len(list_coocurence))
    print(dict_action_embeddings_ELMo.keys() - list_coocurence)


def main():
    # compare_json_files()
    '''
        Annotations
    '''
    clip_length = "3s"
    # path_I3D_features = "../i3d_keras/data/results_features_"+ clip_length + "/"
    #get_clip_time_per_miniclip("../temporal_annotation/miniclips/", "data/dict_clip_time_per_miniclip"+ clip_length  +".json", path_I3D_features, clip_length) # DONE for all channels
    # create_action_clip_labels("data/dict_clip_time_per_miniclip" + clip_length + ".json",
    #                           'data/dict_all_annotations' + clip_length + '.json', channels)

    # '''
    #     Run on LIT1000: preprocess + evaluate Clip I3D features
    # '''
    # os.system("rm -r /local2/oignat/large_data/10s_clips/")
    # create_clips("../temporal_annotation/miniclips/", "/local2/oignat/large_data/10s_clips/", channels[-1])
    # I3D things ..
    # os.system("python /local/oignat/Action_Recog/i3d_keras/src/preprocess.py")
    # run on LIT1000: evaluate_sample.py
    # load_data_from_I3D("../i3d_keras/data/results_features/")


if __name__ == '__main__':
    main()
