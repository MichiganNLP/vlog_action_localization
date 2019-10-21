import glob
import os
import numpy as np

global path_I3D_features
path_I3D_features = "/local/oignat/Action_Recog/keras-kinetics-i3d/data/results_features/"


def create_10s_clips(path_input_video = "/local/oignat/Action_Recog/temporal_annotation/miniclips/"):
    miniclip = path_input_video.split("/")[-1][:-4]
    path_output_video ="/local/oignat/Action_Recog/large_data/10s_clips/"
    if not os.path.exists(path_output_video):
        os.makedirs(path_output_video)

    command = "ffmpeg -i " + path_input_video + " -acodec copy -f segment -segment_time 10 -vcodec copy -reset_timestamps 1 -map 0 " + path_output_video + miniclip+"_%03d.mp4"
    os.system(command)


def create_clips():
    path_input_video = "/local/oignat/Action_Recog/temporal_annotation/miniclips/"
    list_videos = glob.glob(path_input_video + "*.mp4")

    for video_file in list_videos:
        # TODO: Remove this constraint after testing
        miniclip = video_file.split("/")[-1][:-4]
        channel = miniclip.split("_")[0]
        if channel not in ["1p0"]:
            continue
        create_10s_clips(video_file)


def load_data_from_I3D(miniclip="1p0_1mini_1"):
    dict_miniclip_clip_feature = {}
    for filename in os.listdir(path_I3D_features):
        if miniclip in filename:
            features = np.load(path_I3D_features + filename)
            features_mean = features.mean(axis=tuple(range(1, 4)))[0]
            dict_miniclip_clip_feature[filename[:-4]] = features_mean

    features_matrix_I3D = np.zeros((len(dict_miniclip_clip_feature.keys()), 1024))
    index = 0
    for miniclip_clip in sorted(dict_miniclip_clip_feature.keys()):
        I3D_feature_per_clip = dict_miniclip_clip_feature[miniclip_clip]
        features_matrix_I3D[index] = I3D_feature_per_clip
        index += 1

    return features_matrix_I3D


def main():
    path_miniclips = "data/miniclip_actions.json"
    path_pos_data = "data/dict_action_pos_concreteness.json"
    path_list_actions = "data/stats/list_actions.csv"
    path_I3D_features = "/local/oignat/Action_Recog/keras-kinetics-i3d/data/results_features/"
    load_data_from_I3D()


if __name__ == '__main__':
    main()