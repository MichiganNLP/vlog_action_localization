import glob
import json
import os
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from compute_text_embeddings import NumpyEncoder

global path_I3D_features
# on lit1000:  scp results_features/7p* oignat@lit09.eecs.umich.edu:/local/oignat/Action_Recog/i3d_keras/data/results_features/
path_I3D_features = "../i3d_keras/data/results_features/"

def create_10s_clips(path_input_video, path_output_video):
    miniclip = path_input_video.split("/")[-1][:-4]
    # path_output_video = "../large_data/10s_clips/"
    if not os.path.exists(path_output_video):
        os.makedirs(path_output_video)

    command = "ffmpeg -i " + path_input_video + " -acodec copy -f segment -segment_time 10 -vcodec copy -reset_timestamps 1 -map 0 " \
              + path_output_video + miniclip + "_%03d.mp4"
    os.system(command)

# TODO: more exact split ?
def get_clip_time_per_miniclip(path_input, path_output):
    list_videos = glob.glob(path_input + "*.mp4")
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

    with open(path_output, 'w+') as outfile:
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

'''
return -1, if time_1 < time_2
        1, if time_1 > time_2
        0, if equal
'''
def compare_time(time_1, time_2):
    minute_1_s, sec_1_s = time_1.split(":")
    minute_2_s, sec_2_s = time_2.split(":")

    if int(minute_1_s) == int(minute_2_s) and int(sec_1_s) == int(sec_2_s):
        return 0
    if int(minute_1_s) > int(minute_2_s) or (int(minute_1_s) == int(minute_2_s) and int(sec_1_s) >= int(sec_2_s)):
        return 1
    else:
        return -1


def write_clip_length_info():
    # path_clips = "../large_data/10s_clips/"
    path_clips = "/local2/large_data/10s_clips/"
    list_mp4_files = [name.split("/")[-1][:-4] for name in glob.glob(path_clips + "*.mp4")]
    path_output = "data/vlog_movie_length_info.txt"
    with open(path_output, 'a') as the_file:
        for clip in list_mp4_files:
            the_file.write(clip + " " + '10.0' + '\n')


def write_clip_annotations():
    with open('data/time_clips.json') as json_file:
        time_clips = json.load(json_file)

    with open('data/actions_sent_time.json') as json_file:
        actions_sent_time = json.load(json_file)

    clip_time_actions = {}
    for clip in time_clips.keys():
        time_clip = time_clips[clip]
        start_clip_time = time_clip[0]
        end_clip_time = time_clip[1]
        list_actions = []


        for key in actions_sent_time.keys():
            for value in actions_sent_time[key]:
                [time_action, action, transcript] = value
                start_action_time = time_action[0]
                end_action_time = time_action[1]

                # action time equal clip time
                if compare_time(start_clip_time, start_action_time) == 0 and compare_time(end_clip_time,
                                                                                         end_action_time) == 0:
                    list_actions.append(action)

                # action time included in clip time
                elif compare_time(start_clip_time, start_action_time) == -1 and compare_time(end_clip_time,
                                                                                             end_action_time) == 1:
                    list_actions.append(action)

                # clip time included in action time
                elif compare_time(start_action_time, start_clip_time) == -1 and compare_time(end_action_time,
                                                                                             end_clip_time) == 1:
                    list_actions.append(action)

                # action time intersects clip time
                elif compare_time(start_action_time, end_clip_time) == -1 and compare_time(end_action_time, start_clip_time) == 1:
                    list_actions.append(action)

        clip_time_actions[clip] = [time_clip, list_actions]

    for clip in clip_time_actions:
        print(clip, clip_time_actions[clip])


def create_action_clip_labels(path_input, path_output, channels):
    with open(path_input) as f:
        dict_clip_time_per_miniclip = json.loads(f.read())

    all_annotations = []
    dict_all_annotations = {}

    for channel in channels:

        with open("data/annotations/annotations" + channel + ".json") as f:
            dict_annotations = json.loads(f.read())

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
                        and len(range(max(int(time_s_clip), int(time_s_action)), min(int(time_e_clip), int(time_e_action))+1)) > 1: # intersection is > 1s
                        label = True
                    else:
                        label = False
                    # print(miniclip, action, clip, [time_s_clip, time_e_clip], [time_s_action, time_e_action])
                    all_annotations.append([miniclip, clip, action, label])
                    if clip not in dict_all_annotations.keys():
                        dict_all_annotations[clip] = []
                    dict_all_annotations[clip].append([action, label])

    with open(path_output, 'w+') as outfile:
        json.dump(dict_all_annotations, outfile)


def create_clips(path_input_video, path_output_video, channels):
    list_videos = glob.glob(path_input_video + "*.mp4")

    for video_file in list_videos:
        miniclip = video_file.split("/")[-1][:-4]
        channel = miniclip.split("_")[0]
        if channel not in channels:
            continue
        create_10s_clips(video_file, path_output_video)





def save_data_from_I3D():
    max_nb_frames = 100

    dict_miniclip_clip_feature = {}
    for filename in tqdm(os.listdir(path_I3D_features)):
        features = np.load(path_I3D_features + filename)
        # features_mean = features.mean(axis=tuple(range(1, 4)))[0]
        features_per_frame = features.reshape((features.shape[1], features.shape[4]))
        # features_mean = np.zeros(1024)
        # dict_miniclip_clip_feature[filename[:-4]] = features_mean

        padded_video_features = np.zeros((max_nb_frames, 1024))
        for j in range(features.shape[4]):
            if max_nb_frames < features_per_frame.shape[0]:
                raise ValueError(features_per_frame.shape[0])
            padded_video_features[:, j] = np.array(list(features_per_frame[:, j]) + (max_nb_frames - features_per_frame.shape[0]) * [0])

        # dict_miniclip_clip_feature[filename[:-4]] = features_per_frame
        dict_miniclip_clip_feature[filename[:-4]] = padded_video_features

    # features_matrix_I3D = np.zeros((len(dict_miniclip_clip_feature.keys()), 1024))
    # index = 0
    # for miniclip_clip in sorted(dict_miniclip_clip_feature.keys()):
    #     I3D_feature_per_clip = dict_miniclip_clip_feature[miniclip_clip]
    #     features_matrix_I3D[index] = I3D_feature_per_clip
    #     index += 1
    with open('data/dict_I3D_padded.json', 'w+') as outfile:
        json.dump(dict_miniclip_clip_feature, outfile, cls=NumpyEncoder)
    #return dict_miniclip_clip_feature

def load_data_from_I3D():
    # save_data_from_I3D()
    with open('data/dict_I3D_padded.json') as json_file:
        dict_miniclip_clip_feature = json.load(json_file)
    #dict_test_miniclip = {}
    # for key in dict_miniclip_clip_feature.keys():
    #     if "1p0" or "1p1" in key:
    #         dict_test_miniclip[key] = dict_miniclip_clip_feature[key]
    # return dict_test_miniclip
    return dict_miniclip_clip_feature

def average_i3d_features():
    dict_miniclip_clip_feature = {}
    for filename in tqdm(os.listdir(path_I3D_features)):
        features = np.load(path_I3D_features + filename)
        features_mean = features.mean(axis=tuple(range(1, 4)))[0]
        # features_mean = preprocessing.normalize(np.asarray(features_mean).reshape(1,-1), norm='l2')
        # features_mean = np.zeros(1024)
        dict_miniclip_clip_feature[filename[:-4]] = features_mean.reshape(1024)
    return dict_miniclip_clip_feature

def main():
    path_miniclips = "data/miniclip_actions.json"
    path_pos_data = "data/dict_action_pos_concreteness.json"
    path_list_actions = "data/stats/list_actions.csv"
    path_I3D_features = "../i3d_keras/data/results_features/"
    # load_data_from_I3D()
    # get_clip_time_per_miniclip("../temporal_annotation/miniclips/", "data/dict_clip_time_per_miniclip.json")
    create_action_clip_labels("data/dict_clip_time_per_miniclip.json", 'data/dict_all_annotations.json', ["1p0"])


if __name__ == '__main__':
    main()
