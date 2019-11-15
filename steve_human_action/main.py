import codecs
import json
import os

import numpy as np
import glob


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_activity():
    dict = {}
    with open("steve_human_action/activity-list.csv") as fp:
        line = fp.readline()
        while line:
            index = str([int(s) for s in line.split() if s.isdigit()][0])
            if index in line:
                action = line.replace(index, '').strip()
            # print("index {0}, action {1} ".format(index, action))
            line = fp.readline()
            dict[index] = action

    return dict


def read_ouput_DNT():
    dict_embeddings = {}
    list_vec_files = sorted(glob.glob("steve_human_action/output/" + "*.vec"))
    for file_path in list_vec_files:
        with open(file_path, "r") as ins:
            array = []
            for line in ins:
                line = ", ".join(line.split(" ")).replace('\n', '')
                for x in line.split(", "):
                    array.append(float(x))

        index = [str(s) for s in file_path if s.isdigit()]
        string_i = ""
        for s in index:
            string_i += s
        array1 = np.asarray(array)
        if array1.shape != (900,):
            print(array1.shape)
            print(file_path)
        else:
            # raise ValueError("error")
            dict_embeddings[string_i] = array1

    return dict_embeddings


def make_activity_list():
    # read action list from 'dict_actions_cooccurence.json'
    # with open("dict_actions_cooccurence.json") as f:
    #     dict_actions_cooccurence = json.loads(f.read())
    # entire_action = dict_actions_cooccurence["entire_action"]
    # with open("activity-list.csv", "w") as files:
    #     j = 0
    #     for i in list(set(entire_action)):
    #         files.write(str(j) + "\t" + i + "\n")
    #         j += 1

    # read action list from 'dict_action_emb_DNT.json'
    with open('data/dict_action_embeddings_ELMo.json') as json_file:
        data = json.load(json_file)
    with open("steve_human_action/activity-list.csv", "w") as files:
        j = 0
        for i in data.keys():
            files.write(str(j) + "\t" + i + "\n")
            j += 1


def main():
    #make_activity_list()
    '''
    #os.system(" scp steve_human_action/activity-list.csv  oignat@lit1000.eecs.umich.edu:/local2/oignat/activity_prediction-master/clustering/DNT/human_activity_lists")
    # in lit1000:
    # cd /clustering/DNT/
    # python DNT.py --model infersent --dataset activities --dimension 1 --transfer NT --save no --load_model rdnt900.pretrained
    # scp output/*.vec oignat@lit09.eecs.umich.edu:/local/oignat/Action_Recog/action_recog_2/steve_human_action/output/
    '''
    dict_embeddings = read_ouput_DNT()
    dict_actions = read_activity()

    dict_action_emb = {}

    for index in dict_embeddings.keys():
        action = dict_actions[index]
        emb = dict_embeddings[index]
        dict_action_emb[action] = emb.reshape(-1)

    print(len(dict_embeddings.keys()))
    with open('steve_human_action/dict_action_emb_DNT.json', 'w+') as outfile:
        json.dump(dict_action_emb, outfile, cls=NumpyEncoder)


# json.dump(dict_action_emb, codecs.open('dict_action_emb_DNT.json', 'w', encoding='utf-8'), separators=(',', ':'))

if __name__ == '__main__':
    main()
