import json
import math

import numpy as np

from utils_data_text import color


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return iou


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx, :]).astype(np.float)
        this_fp = np.cumsum(fp[tidx, :]).astype(np.float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap


def compute_accuracy_IOU_threshold(threshold, IOU_vals):
    '''
    In this setting, for a given
    value of threshold, whenever a given predicted time window has
    an intersection with the gold-standard that is above the α
    threshold, we consider the output of the model as correct.
    :param threshold: 0.1, 0.3, 0.5, 0.7
    '''
    nb_correct = 0
    nb_total = len(IOU_vals)
    for tIOU in IOU_vals:
        if max(tIOU) > threshold:
            nb_correct += 1
    tIOU_threshold = nb_correct / nb_total * 100
    print("tIOU@{:.1} = {:.2f}".format(threshold, tIOU_threshold))
    return tIOU_threshold


def compute_meanIOU(IOU_vals):
    mean_tIOU = np.nanmean([max(x) for x in IOU_vals])
    mean_tIOU = mean_tIOU * 100
    print(color.PURPLE + color.BOLD + "mIOU" + color.END + " = {:.2f}".format(mean_tIOU))
    return mean_tIOU


def wrapper_IOU_combine_2(predicted_time, proposed_1p0_1, proposed_1p0_2, groundtruth_1p0):
    if len(proposed_1p0_1.keys()) != len(groundtruth_1p0.keys()):
        count_visible_actions_not_caught = 0
        for key in groundtruth_1p0.keys() - proposed_1p0_1.keys():
            if groundtruth_1p0[key] != ['not visible']:
                count_visible_actions_not_caught += 1
                # print(key)
        if count_visible_actions_not_caught:
            print("count_visible_actions_not_caught: " + str(count_visible_actions_not_caught))

    if len(proposed_1p0_2.keys()) != len(groundtruth_1p0.keys()):
        count_visible_actions_not_caught = 0
        for key in groundtruth_1p0.keys() - proposed_1p0_2.keys():
            if groundtruth_1p0[key] != ['not visible']:
                count_visible_actions_not_caught += 1
                # print(key)
        if count_visible_actions_not_caught:
            print("count_visible_actions_not_caught: " + str(count_visible_actions_not_caught))

    IOU_vals = []
    dict_IOU_per_length = {}
    for miniclip_action in groundtruth_1p0.keys():
        # TODO: deal with this
        if groundtruth_1p0[miniclip_action] == ['not visible']:
            continue
        if miniclip_action not in proposed_1p0_1.keys():
            continue
        if proposed_1p0_1[miniclip_action] == ['not visible']:
            continue
        if miniclip_action not in proposed_1p0_2.keys():
            continue
        if proposed_1p0_2[miniclip_action] == ['not visible']:
            continue

        target_segment = np.array([float(x) for x in groundtruth_1p0[miniclip_action]])

        action_duration = target_segment[1] - target_segment[0]
        rounded_duration = str(int(round(action_duration, -1)))
        # if rounded_duration in ['0', '10']:
        #     candidate_segments = np.array(proposed_1p0_1[miniclip_action])  # alignemnt is good for short actions
        # else:
        #     candidate_segments = np.array(proposed_1p0_2[miniclip_action]) # MPU is good for long actions

        predicted_duration = int(predicted_time[miniclip_action.split(", ")[1]])

        if predicted_duration == 1:
            candidate_segments = np.array(proposed_1p0_1[miniclip_action])  # alignment is good for short actions
        else:
            candidate_segments = np.array(proposed_1p0_2[miniclip_action])  # MPU is good for long actions

        # elif rounded_duration in ['40', '50', '60']:
        #     candidate_segments = np.array(
        #         proposed_1p0_3[miniclip_action])  # cosine action I3D + BERT is good for long actions
        # else:
        #     candidate_segments = np.array(proposed_1p0_2[miniclip_action])  # MPU is good for medium actions

        tIOU = segment_iou(target_segment, candidate_segments)
        # tIOU = np.expand_dims(tIOU, axis=1)
        # IOU_vals = np.append(IOU_vals, tIOU, axis=0)
        IOU_vals.append(tIOU)

        if rounded_duration not in dict_IOU_per_length.keys():
            dict_IOU_per_length[rounded_duration] = []
        dict_IOU_per_length[rounded_duration].append(tIOU)

    return IOU_vals, dict_IOU_per_length


def wrapper_IOU(proposed_1p0, groundtruth_1p0):
    if len(proposed_1p0.keys()) != len(groundtruth_1p0.keys()):
        count_visible_actions_not_caught = 0
        for key in groundtruth_1p0.keys() - proposed_1p0.keys():
            if groundtruth_1p0[key] != ['not visible']:
                count_visible_actions_not_caught += 1
                # print(key)
        if count_visible_actions_not_caught:
            print("count_visible_actions_not_caught: " + str(count_visible_actions_not_caught))

    IOU_vals = []
    dict_IOU_per_length = {}
    dict_IOU_per_position = {'10': [], '50': []}
    gt = len(groundtruth_1p0.keys())
    p = len(proposed_1p0.keys())
    print(gt)
    print(p)
    nb_not_vis_total = 0
    nb_not_vis = 0

    for miniclip_action in groundtruth_1p0.keys():
        # TODO: deal with this
        if groundtruth_1p0[miniclip_action][0] == 'not visible':
            groundtruth_1p0[miniclip_action] = [-1, -1]
            nb_not_vis_total += 1
            # continue
        if miniclip_action not in proposed_1p0.keys():
            continue
        # if proposed_1p0[miniclip_action] == ['not visible']:
        #     continue

        target_segment = np.array([float(x) for x in groundtruth_1p0[miniclip_action]])
        candidate_segments = np.array(proposed_1p0[miniclip_action])

        tIOU = segment_iou(target_segment, candidate_segments)
        if math.isnan(tIOU[0]):
            tIOU = [1]
            nb_not_vis += 1
            if groundtruth_1p0[miniclip_action] != [-1, -1] or proposed_1p0[miniclip_action][0] != [-1, -1]:
                print("NOOOO: ")
                print(groundtruth_1p0[miniclip_action])
                print(proposed_1p0[miniclip_action])
                break

        # tIOU = np.expand_dims(tIOU, axis=1)
        # IOU_vals = np.append(IOU_vals, tIOU, axis=0)
        IOU_vals.append(tIOU)

        action_duration = target_segment[1] - target_segment[0]
        rounded_duration = str(int(round(action_duration, -1)))
        if rounded_duration == "10":
            rounded_duration = "0"
        elif rounded_duration == "30":
            rounded_duration = "20"
        elif rounded_duration in ["40", "50", "60"]:
            rounded_duration = "50"

        if rounded_duration not in dict_IOU_per_length.keys():
            dict_IOU_per_length[rounded_duration] = []
        # if rounded_duration == "60":
        #     print(str(target_segment) + ", " + str(candidate_segments))
        dict_IOU_per_length[rounded_duration].append(tIOU)

        if target_segment[0] <= 10:
            dict_IOU_per_position["10"].append(tIOU)
        else:
            dict_IOU_per_position["50"].append(tIOU)

    print("# not visible predicted as not visible " + str(nb_not_vis) + "out of #GT " + str(nb_not_vis_total))
    return IOU_vals, dict_IOU_per_length, dict_IOU_per_position


# def compute_recall_IOU_threshold(threshold, IOU_vals, nb_proposals):
#
#     nb_correct = 0
#     nb_total = len(IOU_vals)
#     for tIOU in IOU_vals:
#         if tIOU > threshold:
#             nb_correct += 1
#     tIOU_threshold = nb_correct / nb_total
#     print("Accuracy tIOU = {:.2f} for threshold = {:.1}".format(tIOU_threshold, threshold))
#     return tIOU_threshold


def evaluate_combine_2(predicted_time, method1, method2, channel):
    print("-----------------------------------------------------------")
    print("Results for method {0}, {1} on channel {2}:".format(method1, method2, channel))
    with open("data/results/dict_predicted_" + method1 + ".json") as f:
        proposed_1p0_1 = json.loads(f.read())

    with open("data/results/dict_predicted_" + method2 + ".json") as f:
        proposed_1p0_2 = json.loads(f.read())

    with open("data/annotations/annotations" + channel + ".json") as f:
        groundtruth_1p0 = json.loads(f.read())

    IOU_vals, dict_IOU_per_length = wrapper_IOU_combine_2(predicted_time, proposed_1p0_1, proposed_1p0_2,
                                                          groundtruth_1p0)
    print("#test points: " + str(len(IOU_vals)))

    list_results = []
    for threshold in np.arange(0.1, 0.9, 0.2):
        accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
        list_results.append(str(round(accuracy, 2)))

    mean_tIOU = compute_meanIOU(IOU_vals)
    list_results.append(str(round(mean_tIOU, 2)))
    print(color.GREEN + color.BOLD + "overleaf: " + color.END + list_results[0] + " & " + list_results[1] + " & " +
          list_results[2] + " & " + list_results[
              3] + " & " + list_results[4])

    for action_duration in dict_IOU_per_length.keys():
        IOU_vals = dict_IOU_per_length[action_duration]
        print(action_duration)
        print("#test points: " + str(len(IOU_vals)))

        list_results = []
        for threshold in np.arange(0.1, 0.9, 0.2):
            accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
            list_results.append(str(round(accuracy, 2)))

        mean_tIOU = compute_meanIOU(IOU_vals)
        list_results.append(str(round(mean_tIOU, 2)))
        print(color.GREEN + color.BOLD + "overleaf: " + color.END + list_results[0] + " & " + list_results[1] + " & " +
              list_results[2] + " & " + list_results[
                  3] + " & " + list_results[4])


def evaluate(method, channel):
    print("-----------------------------------------------------------")
    print("Results for method {0} on channel {1}:".format(method, channel))
    with open("data/results/dict_predicted_" + method + ".json") as f:
        proposed_1p0 = json.loads(f.read())

    with open("data/annotations/new/annotations" + channel + ".json") as f:
        groundtruth_1p0 = json.loads(f.read())

    # with open("data/annotations/annotations1p01_5p01_vb.json") as f:
    #     groundtruth_1p0 = json.loads(f.read())

    IOU_vals, dict_IOU_per_length, dict_IOU_per_position = wrapper_IOU(proposed_1p0, groundtruth_1p0)
    print("#test points: " + str(len(IOU_vals)))

    list_results = []
    for threshold in np.arange(0.1, 0.9, 0.2):
        accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
        list_results.append(str(round(accuracy, 2)))

    mean_tIOU = compute_meanIOU(IOU_vals)
    list_results.append(str(round(mean_tIOU, 2)))
    print(color.GREEN + color.BOLD + "overleaf: " + color.END + list_results[0] + " & " + list_results[1] + " & " +
          list_results[2] + " & " + list_results[
              3] + " & " + list_results[4])

    for action_duration in dict_IOU_per_length.keys():
        IOU_vals = dict_IOU_per_length[action_duration]
        print(action_duration)
        print("#test points: " + str(len(IOU_vals)))

        list_results = []
        for threshold in np.arange(0.1, 0.9, 0.2):
            accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
            list_results.append(str(round(accuracy, 2)))

        mean_tIOU = compute_meanIOU(IOU_vals)
        list_results.append(str(round(mean_tIOU, 2)))
        print(color.GREEN + color.BOLD + "overleaf: " + color.END + list_results[0] + " & " + list_results[1] + " & " +
              list_results[2] + " & " + list_results[
                  3] + " & " + list_results[4])

    print("------------- IOU by action pos ------")
    for action_position in dict_IOU_per_position.keys():
        IOU_vals = dict_IOU_per_position[action_position]
        print(action_position)
        print("#test points: " + str(len(IOU_vals)))

        list_results = []
        for threshold in np.arange(0.1, 0.9, 0.2):
            accuracy = compute_accuracy_IOU_threshold(threshold, IOU_vals)
            list_results.append(str(round(accuracy, 2)))

        mean_tIOU = compute_meanIOU(IOU_vals)
        list_results.append(str(round(mean_tIOU, 2)))
        print(color.GREEN + color.BOLD + "overleaf: " + color.END + list_results[0] + " & " + list_results[1] + " & " +
              list_results[2] + " & " + list_results[
                  3] + " & " + list_results[4])


if __name__ == "__main__":
    # with open("data/stemmed_actions_miniclip_time1p0.json") as f:
    #     proposed_1p0 = json.loads(f.read())

    # for channel in ["1p0", "1p1", "2p0", "2p1", "3p0", "3p1", "4p0", "4p1", "5p0", "5p1"]:
    channel = "1p0"
    with open("data/results/dict_predicted_cosine sim_ELMo" + channel + ".json") as f:
        proposed_1p0 = json.loads(f.read())

    # with open("data/annotations/annotations5p1.json") as f:
    #     proposed_1p0 = json.loads(f.read())

    # with open("data/dict_predicted_1p0.json") as f:
    #     proposed_1p0 = json.loads(f.read())

    with open("data/annotations/annotations" + channel + ".json") as f:
        groundtruth_1p0 = json.loads(f.read())

    # with open("data/annotations/yuhang_data/results5p1_tonyzhou.json") as f:
    #     groundtruth_1p0 = json.loads(f.read())

    IOU_vals = wrapper_IOU(proposed_1p0, groundtruth_1p0)

    # threshold: [0.1, 0.3, 0.5, 0.7]
    for threshold in np.arange(0.1, 0.9, 0.2):
        compute_accuracy_IOU_threshold(threshold, IOU_vals)
    compute_meanIOU(IOU_vals)
