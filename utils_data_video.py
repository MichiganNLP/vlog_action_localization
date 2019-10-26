import glob
import json
import os
import numpy as np

global path_I3D_features
path_I3D_features = "../i3d_keras/data/results_features/"


def create_10s_clips(path_input_video="../temporal_annotation/miniclips/"):
    miniclip = path_input_video.split("/")[-1][:-4]
    path_output_video = "../large_data/10s_clips/"
    if not os.path.exists(path_output_video):
        os.makedirs(path_output_video)

    command = "ffmpeg -i " + path_input_video + " -acodec copy -f segment -segment_time 10 -vcodec copy -reset_timestamps 1 -map 0 " \
              + path_output_video + miniclip + "_%03d.mp4"
    os.system(command)

# TODO: more exact split ?
def get_clip_time_per_miniclip():
    path_input_video = "../temporal_annotation/miniclips/"
    list_videos = glob.glob(path_input_video + "*.mp4")
    dict_clip_time_per_miniclip = {}

    for video_file in list_videos:
        clip_name_root = video_file.split("/")[-1][:-4]
        show_sec_time = "ffprobe -i " + video_file + " -show_entries format=duration -v quiet -of csv=p=0"
        time_miniclip = float(os.popen(show_sec_time).read())
        nb_clips = np.math.ceil(time_miniclip / 10)  # for 10s clips
        for index in range(nb_clips):
            clip_name = clip_name_root + "_" + str(index).zfill(3) + ".mp4"
            if index == 0:
                start_time = 0
                if time_miniclip >= 10:
                    end_time = 10
                else:
                    end_time = 10 - time_miniclip
            else:
                start_time = index * 10
                if time_miniclip >= 10:
                    end_time = start_time + 10
                else:
                    end_time = start_time + time_miniclip

            time_miniclip = time_miniclip - 10
            dict_clip_time_per_miniclip[clip_name] = [start_time, end_time]

    with open('data/dict_clip_time_per_miniclip.json', 'w+') as outfile:
        json.dump(dict_clip_time_per_miniclip, outfile)


def get_all_keys_for_substring(substring, dict):
    list_keys = []
    for key in dict.keys():
        if substring in key:
            list_keys.append(key)
    return list_keys


def intervals_overlap(i1, i2):
    [x1, x2] = i1
    [y1, y2] = i2
    return x1 <= y2 and y1 <= x2


# TODO: ALL Annotations, just 1p0 now
def create_action_clip_labels():
    with open("data/dict_clip_time_per_miniclip.json") as f:
        dict_clip_time_per_miniclip = json.loads(f.read())

    with open("data/annotations/annotations1p0.json") as f:
        dict_annotations = json.loads(f.read())

    all_annotations = []
    dict_all_annotations = {}

    for miniclip_action in dict_annotations.keys():
        if dict_annotations[miniclip_action] != ["not visible"]:
            [time_s, time_e] = dict_annotations[miniclip_action]
            time_s_action = float(time_s)
            time_e_action = float(time_e)
            miniclip = miniclip_action.split(", ")[0]
            action = miniclip_action.split(", ")[1]
            list_clips = get_all_keys_for_substring(miniclip[:-4] + "_", dict_clip_time_per_miniclip)
            for clip in list_clips:
                [time_s_clip, time_e_clip] = dict_clip_time_per_miniclip[clip]
                if intervals_overlap([time_s_clip, time_e_clip], [time_s_action, time_e_action]) \
                    and len(range(max(int(time_s_clip), int(time_s_action)), min(int(time_e_clip), int(time_e_action))+1)) > 3: # intersection is > 3s
                    label = True
                else:
                    label = False
                # print(miniclip, action, clip, [time_s_clip, time_e_clip], [time_s_action, time_e_action])
                all_annotations.append([miniclip, clip, action, label])
                if clip not in dict_all_annotations.keys():
                    dict_all_annotations[clip] = []
                dict_all_annotations[clip].append([action, label])

    with open('data/dict_all_annotations.json', 'w+') as outfile:
        json.dump(dict_all_annotations, outfile)


def create_clips():
    path_input_video = "../temporal_annotation/miniclips/"
    list_videos = glob.glob(path_input_video + "*.mp4")

    for video_file in list_videos:
        # TODO: Remove this constraint after testing
        miniclip = video_file.split("/")[-1][:-4]
        channel = miniclip.split("_")[0]
        if channel not in ["1p0"]:
            continue
        create_10s_clips(video_file)
        break


def load_data_from_I3D():
    dict_miniclip_clip_feature = {}
    for filename in os.listdir(path_I3D_features):
        features = np.load(path_I3D_features + filename)
        features_mean = features.mean(axis=tuple(range(1, 4)))[0]
        dict_miniclip_clip_feature[filename[:-4]] = features_mean

    # features_matrix_I3D = np.zeros((len(dict_miniclip_clip_feature.keys()), 1024))
    # index = 0
    # for miniclip_clip in sorted(dict_miniclip_clip_feature.keys()):
    #     I3D_feature_per_clip = dict_miniclip_clip_feature[miniclip_clip]
    #     features_matrix_I3D[index] = I3D_feature_per_clip
    #     index += 1

    return dict_miniclip_clip_feature


def main():
    path_miniclips = "data/miniclip_actions.json"
    path_pos_data = "data/dict_action_pos_concreteness.json"
    path_list_actions = "data/stats/list_actions.csv"
    path_I3D_features = "../i3d_keras/data/results_features/"
    # load_data_from_I3D()
    # get_clip_time_per_miniclip()
    create_action_clip_labels()


if __name__ == '__main__':
    main()
