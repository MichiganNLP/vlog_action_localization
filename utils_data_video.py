import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from tqdm import tqdm

from compute_text_embeddings import NumpyEncoder, create_bert_embeddings, save_elmo_embddings


# global path_I3D_features
# on lit1000:  scp results_features/7p* oignat@lit09.eecs.umich.edu:/local/oignat/Action_Recog/i3d_keras/data/results_features/

def create_10s_clips(path_input_video, path_output_video):
    miniclip = path_input_video.split("/")[-1][:-4]
    # path_output_video = "../large_data/10s_clips/"
    if not os.path.exists(path_output_video):
        os.makedirs(path_output_video)

    command = "ffmpeg -i " + path_input_video + " -acodec copy -f segment -segment_time 10 -vcodec copy -reset_timestamps 1 -map 0 " \
              + path_output_video + miniclip + "_%03d.mp4"
    os.system(command)


# TODO: more exact split ?
def get_clip_time_per_miniclip(path_input, path_output, path_I3D_features, clip_length):
    list_videos = glob.glob(path_input + "*.mp4")
    dict_clip_time_per_miniclip = {}
    clip_length = int(clip_length[0])

    for video_file in list_videos:
        clip_name_root = video_file.split("/")[-1][:-4]
        show_sec_time = "ffprobe -i " + video_file + " -show_entries format=duration -v quiet -of csv=p=0"
        time_miniclip = float(os.popen(show_sec_time).read())
        # nb_clips = np.math.ceil(time_miniclip / clip_length)  # for clip_length clips
        nb_clips = int(time_miniclip)  # for clip_length clips
        for index in range(nb_clips):
            clip_name = clip_name_root + "_" + str(index).zfill(3) + ".mp4"
            if index == 0:
                start_time = 0
                if time_miniclip >= clip_length:
                    end_time = clip_length
                else:
                    end_time = clip_length - time_miniclip
            else:
                # start_time = index * clip_length # if non-overlapping
                start_time = index - 3 + clip_length
                if time_miniclip >= clip_length:
                    end_time = start_time + clip_length
                else:
                    end_time = start_time + time_miniclip

            # time_miniclip = time_miniclip - clip_length
            time_miniclip = time_miniclip - 1
            dict_clip_time_per_miniclip[clip_name] = [start_time, end_time]

    list_i3d_files = []
    for filename in tqdm(os.listdir(path_I3D_features)):
        list_i3d_files.append(filename[:-4] + ".mp4")

    new_dict_clip_time_per_miniclip = dict_clip_time_per_miniclip.copy()

    for key in dict_clip_time_per_miniclip.keys():
        if key not in list_i3d_files:
            del new_dict_clip_time_per_miniclip[key]

    with open(path_output, 'w+') as outfile:
        json.dump(new_dict_clip_time_per_miniclip, outfile)


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
                elif compare_time(start_action_time, end_clip_time) == -1 and compare_time(end_action_time,
                                                                                           start_clip_time) == 1:
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
                            and len(range(max(int(time_s_clip), int(time_s_action)),
                                          min(int(time_e_clip), int(time_e_action)) + 1)) > 1:  # intersection is > 1s
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


def save_data_from_I3D(path_I3D_features):
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
            print(features_per_frame.shape[0])
            if max_nb_frames < features_per_frame.shape[0]:
                raise ValueError(features_per_frame.shape[0])
            padded_video_features[:, j] = np.array(
                list(features_per_frame[:, j]) + (max_nb_frames - features_per_frame.shape[0]) * [0])

        # dict_miniclip_clip_feature[filename[:-4]] = features_per_frame
        dict_miniclip_clip_feature[filename[:-4]] = padded_video_features

    # features_matrix_I3D = np.zeros((len(dict_miniclip_clip_feature.keys()), 1024))
    # index = 0
    # for miniclip_clip in sorted(dict_miniclip_clip_feature.keys()):
    #     I3D_feature_per_clip = dict_miniclip_clip_feature[miniclip_clip]
    #     features_matrix_I3D[index] = I3D_feature_per_clip
    #     index += 1
    with open('data/embeddings/dict_I3D_padded.json', 'w+') as outfile:
        json.dump(dict_miniclip_clip_feature, outfile, cls=NumpyEncoder)
    # return dict_miniclip_clip_feature


def load_data_from_I3D(path_I3D_features):
    save_data_from_I3D(path_I3D_features)
    with open('data/embeddings/dict_I3D_padded.json') as json_file:
        dict_miniclip_clip_feature = json.load(json_file)
    print(len(dict_miniclip_clip_feature.keys()))
    # dict_test_miniclip = {}
    # for key in dict_miniclip_clip_feature.keys():
    #     if "1p0" or "1p1" in key:
    #         dict_test_miniclip[key] = dict_miniclip_clip_feature[key]
    # return dict_test_miniclip
    return dict_miniclip_clip_feature


def average_i3d_features(path_I3D_features):
    with open('data/embeddings/dict_I3D_avg.json') as json_file:
        dict_clip_feature = json.load(json_file)
    # print(len(dict_clip_feature.keys()))

    # dict_clip_feature = {}
    # print("loading I3D")
    # for filename in tqdm(os.listdir(path_I3D_features)):
    #     features = np.load(path_I3D_features + filename)
    #     features_mean = features.mean(axis=tuple(range(1, 4)))[0]
    #     # features_mean = preprocessing.normalize(np.asarray(features_mean).reshape(1,-1), norm='l2')
    #     # features_mean = np.zeros(1024)
    #     dict_clip_feature[filename[:-4]] = features_mean.reshape(1024)
    #
    # with open('data/embeddings/dict_I3D_avg.json', 'w+') as outfile:
    #     json.dump(dict_clip_feature, outfile, cls=NumpyEncoder)

    return dict_clip_feature


def average_i3d_features_miniclip(path_I3D_features):
    # with open('data/embeddings/dict_I3D_avg_miniclip.json') as json_file:
    #     dict_miniclip_feature = json.load(json_file)
    # print(len(dict_miniclip_feature.keys()))

    dict_miniclip_feature = {}
    print("loading I3D")
    for filename in tqdm(os.listdir(path_I3D_features)):
        features = np.load(path_I3D_features + filename)
        features_mean = features.mean(axis=tuple(range(1, 4)))[0]
        # features_mean = preprocessing.normalize(np.asarray(features_mean).reshape(1,-1), norm='l2')
        # features_mean = np.zeros(1024)
        dict_miniclip_feature[filename[:-4]] = features_mean.reshape(1024)

    with open('data/embeddings/dict_I3D_avg_miniclip.json', 'w+') as outfile:
        json.dump(dict_miniclip_feature, outfile, cls=NumpyEncoder)

    return dict_miniclip_feature


def load_FasterRCNN_feat():
    # path_feat = "../FasterRCNN/processed/"
    path_feat = "/local2/jiajunb/data/processed/"
    dict_FasterRCNN_original = {}
    for miniclip in tqdm(os.listdir(path_feat)):
        dict_FasterRCNN_original[miniclip] = {}
        for frame in os.listdir(path_feat + miniclip + "/"):
            dict_FasterRCNN_original[miniclip][frame[:-7]] = []
            root = Path(path_feat + miniclip + "/" + frame)
            tensor = torch.load(root, map_location="cpu")  # add map_location here; otherwise, it will map to gpu

            bbox_label = tensor[0].pred_classes.numpy()
            if bbox_label.size == 0:
                dict_FasterRCNN_original[miniclip][frame[:-7]] = np.array([-1])
                continue
            # elif bbox_label.size >= 3:
            #     bbox_label_first = list(tensor[0].pred_classes.numpy())[0:3]
            # else:
            #     bbox_label_first = list(tensor[0].pred_classes.numpy())
            # print(np.array(bbox_label_first))
            # dict_FasterRCNN_original[miniclip][frame[:-7]] = np.array(bbox_label_first)
            bbox_score = tensor[0].scores.numpy()
            bbox_features = tensor[1].numpy()
            # bbox_features_list = []
            # for i, score in enumerate(list(bbox_score)):
            #     if score > 0.5:
            #         bbox_features_list.append(bbox_features)

            # dict_FasterRCNN_original[miniclip][frame[:-7]] = np.array(bbox_features)

            # dict_FasterRCNN_original[miniclip][frame[:-7]]['label'] = bbox_label
            dict_FasterRCNN_original[miniclip][frame[:-7]]['score'] = bbox_score
            dict_FasterRCNN_original[miniclip][frame[:-7]]['features']= np.array(bbox_features)

    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_original_bbox_features.json', 'w+') as outfile:
        json.dump(dict_FasterRCNN_original, outfile, cls=NumpyEncoder)


def read_FasterRCNN():
    # import metadata catalog class
    from detectron2.data import MetadataCatalog
    # get meta data
    MetadataCatalog.get('coco_2017_train')
    # get the list of thing
    list_classes = MetadataCatalog.get('coco_2017_train').thing_classes
    dict_FasterRCNN_first_label = {}
    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_original_first3_label.json') as json_file:
        dict_FasterRCNN_original = json.load(json_file)

    for miniclip in tqdm(list(dict_FasterRCNN_original.keys())):
        dict_FasterRCNN_first_label[miniclip] = {}
        for frame in dict_FasterRCNN_original[miniclip].keys():
            index_bbox_label_first = list(dict_FasterRCNN_original[miniclip][frame])  # from np.array([x]) to [x]

            dict_FasterRCNN_first_label[miniclip][frame] = []
            for i in range(len(index_bbox_label_first)):
                if index_bbox_label_first[i] == -1:
                    label_text = "none"
                else:
                    label_text = list_classes[index_bbox_label_first[i]]

                dict_FasterRCNN_first_label[miniclip][frame].append(label_text)

    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_first_label_str.json', 'w+') as outfile:
        json.dump(dict_FasterRCNN_first_label, outfile)


def transform_miniclip_data_into_clips():
    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_first_label_str.json') as json_file:
        dict_FasterRCNN_first_label_str = json.load(json_file)

    dict_clips_data = {}
    # set_classes = set()
    for miniclip in tqdm(list(dict_FasterRCNN_first_label_str.keys())):
        nb_frames = len(dict_FasterRCNN_first_label_str[miniclip].keys())

        list_classes_miniclip = []
        for frame in sorted(dict_FasterRCNN_first_label_str[miniclip].keys()):
            class_name = dict_FasterRCNN_first_label_str[miniclip][frame]
            for c in class_name:
                if "person" not in c:
                    list_classes_miniclip.append(c)
            # print(frame, str(frame_nb), class_name)

        for index_clip in range(0, int((nb_frames - 72) / 24)):
            clip_name = miniclip + "_" + str(index_clip).zfill(3)
            if clip_name not in dict_clips_data.keys():
                dict_clips_data[clip_name] = []

            # dict_clips_data[clip_name] = list(set(list_classes_miniclip[index_clip * 24:(index_clip + 3) * 24]))
            dict_clips_data[clip_name] = list_classes_miniclip[index_clip * 24:(index_clip + 3) * 24]

    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_first3_label_str_clips.json', 'w+') as outfile:
        json.dump(dict_clips_data, outfile)


def transform_miniclip_data_into_clips_dandan():
    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_dandan_str.json') as json_file:
        dict_FasterRCNN_dadan_str = json.load(json_file)

    dict_clips_data = {}
    set_classes = set()
    for miniclip in tqdm(list(dict_FasterRCNN_dadan_str.keys())):
        nb_frames = len(dict_FasterRCNN_dadan_str[miniclip].keys())

        list_classes_miniclip = []
        for frame in sorted(dict_FasterRCNN_dadan_str[miniclip].keys()):
            class_name = [c for [c, score] in dict_FasterRCNN_dadan_str[miniclip][frame] if score > 0.5]
            for c in class_name:
                if "_" in c:
                    action = " ".join(c.split("_")).lower()
                else:
                    action = c.lower()
                if "wheel" not in action:
                    list_classes_miniclip.append(action)

                set_classes.add(action)
            # print(frame, str(frame_nb), class_name)

        for index_clip in range(0, int((nb_frames - 72) / 24)):
            clip_name = miniclip + "_" + str(index_clip).zfill(3)
            if clip_name not in dict_clips_data.keys():
                dict_clips_data[clip_name] = []

            # dict_clips_data[clip_name] = list(set(list_classes_miniclip[index_clip * 24:(index_clip + 3) * 24:10]))
            dict_clips_data[clip_name] = list_classes_miniclip[index_clip * 24:(index_clip + 3) * 24]

    # create_bert_embeddings(list(set_classes))
    # save_elmo_embddings(list(set_classes))

    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_dandan_str_clips.json', 'w+') as outfile:
        json.dump(dict_clips_data, outfile)


def get_feature_and_label(resnet50_feature, resnet50_label, preprocess, class2name_mapping, image, bbox):
    # preprocess
    input_image = image.crop(bbox)  # bbox=(left, upper, right, lower)
    input_image = preprocess(input_image).unsqueeze(0).cuda()

    # predict
    feature = resnet50_feature(input_image).view(1, -1)  # should 1x2048
    predicted_prob = resnet50_label(input_image)
    predicted_label = torch.max(predicted_prob.cpu(), dim=1)[1].item()  # int
    predicted_name = class2name_mapping[predicted_label]

    return feature, predicted_label, predicted_name


def read_data_DanDan():
    import os, pickle, glob, json
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import transforms
    from cnn_finetune import make_model

    # pretrained resnet-50
    resnet50 = models.resnet50(pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50_feature = nn.Sequential(*modules)
    resnet50_feature.eval()

    # our finetuned resnet-50, object classifier
    model_path = 'data/FasterRCNN/resnet50_object_lr0.001acc0.6587dp0.2_epoch19.pth'
    resnet50_label = make_model('resnet50', num_classes=423)
    resnet50_label.load_state_dict(torch.load(model_path))
    resnet50_label.eval()

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm])

    # to cuda
    resnet50_feature.cuda()
    resnet50_label.cuda()

    # class2label mapping
    with open('data/FasterRCNN/class2label.json', 'r') as f:
        name2class_mapping = json.load(f)
        class2name_mapping = {}
        for k, v in name2class_mapping.items():
            class2name_mapping[v] = k

    # ------------------------------------------------------------------------------------------------

    # read pkl
    result_list = glob.glob('data/FasterRCNN/FasterRCNN_dandan/*.pkl')
    dict_FasterRCNN_dandan = {}
    for i in tqdm(range(len(result_list))):
        # for i in tqdm(range(len([1,2]))):
        with open(result_list[i], 'rb') as f:
            prediction = pickle.load(f)

            for i, (image_path, val) in enumerate(prediction.items()):
                # if i % 10 != 0: continue
                image_path = "../i3d_keras/data/frames/" + "/".join(image_path.split("/")[-2:])

                image_folder, image_name = os.path.split(image_path)
                miniclip = image_folder.split("/")[-1]
                frame = image_name[:-4]
                if miniclip not in dict_FasterRCNN_dandan.keys():
                    dict_FasterRCNN_dandan[miniclip] = {}
                if frame not in dict_FasterRCNN_dandan[miniclip].keys():
                    dict_FasterRCNN_dandan[miniclip][frame] = {'bbox_score':[],'bbox_names':[], 'bbox_features':[]}
                    # dict_FasterRCNN_dandan[miniclip][frame] = []

                if 'object_info' in val.keys():
                    object_info = val['object_info']

                    image = Image.open(image_path)
                    draw = ImageDraw.Draw(image)
                    for bbox_index, bbox_info in object_info.items():
                        bbox = (
                            bbox_info['bbox']['x1'], bbox_info['bbox']['y1'], bbox_info['bbox']['x2'],
                            bbox_info['bbox']['y2'])
                        score = bbox_info['score']

                        feature, predicted_label, predicted_name = get_feature_and_label(resnet50_feature,
                                                                                         resnet50_label,
                                                                                         preprocess, class2name_mapping,
                                                                                         image, bbox)
                        # print(feature.shape)
                        # print(image_folder, image_name)
                        # print(predicted_name)
                        dict_FasterRCNN_dandan[miniclip][frame]['bbox_score'].append(score)
                        # dict_FasterRCNN_dandan[miniclip][frame]['bbox_features'].append(feature.cpu().numpy())
                        dict_FasterRCNN_dandan[miniclip][frame]['bbox_features'].append(feature.cpu().detach().numpy())
                        dict_FasterRCNN_dandan[miniclip][frame]['bbox_names'].append(predicted_name)
                        # dict_FasterRCNN_dandan[miniclip][frame].append(score, predicted_name, feature)
                        # dict_FasterRCNN_dandan[miniclip][frame].append((score, predicted_name))
                        # print(dict_FasterRCNN_dandan)


    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_dandan_all.json', 'w+') as outfile:
        json.dump(dict_FasterRCNN_dandan, outfile)


def transform_clip_to_frames():
    with open('data/embeddings/FasterRCNN/dict_FasterRCNN_first_label_str.json') as json_file:
        dict_FasterRCNN_first_label_str = json.load(json_file)

    dict_clips_data = {}
    # set_classes = set()
    for miniclip in tqdm(list(dict_FasterRCNN_first_label_str.keys())):
        nb_frames = len(dict_FasterRCNN_first_label_str[miniclip].keys())

        for index_clip in range(0, int((nb_frames - 72) / 24)):
            clip_name = miniclip + "_" + str(index_clip+1).zfill(3)

            if clip_name not in dict_clips_data.keys():
                dict_clips_data[clip_name] = []

            for index_frame in range(index_clip * 24, (index_clip + 3) * 24 + 1):
                if index_frame == 0:
                    continue
                frame_name = "frame_" + str(index_frame).zfill(5)
                dict_clips_data[clip_name].append(frame_name)



    with open('data/embeddings/clip_to_frames.json', 'w+') as outfile:
        json.dump(dict_clips_data, outfile)


def main():
    path_miniclips = "data/miniclip_actions.json"
    path_pos_data = "data/dict_action_pos_concreteness.json"
    path_list_actions = "data/stats/list_actions.csv"

    load_FasterRCNN_feat()
    # read_FasterRCNN()
    # transform_miniclip_data_into_clips()
    # read_data_DanDan()
    # transform_miniclip_data_into_clips_dandan()

    # transform_clip_to_frames()
    # path_I3D_features = "../i3d_keras/data/results_features/"
    # load_data_from_I3D()
    # get_clip_time_per_miniclip("../temporal_annotation/miniclips/", "data/dict_clip_time_per_miniclip.json")
    # create_action_clip_labels("data/dict_clip_time_per_miniclip.json", 'data/dict_all_annotations.json', ["1p0"])


if __name__ == '__main__':
    main()
