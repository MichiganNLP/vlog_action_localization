import glob
import json

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
    path_clips = "../large_data/10s_clips/"
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

def main():
    write_clip_annotations()


if __name__ == '__main__':
    main()
