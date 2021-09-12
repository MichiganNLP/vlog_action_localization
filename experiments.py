from args import parse_args, channels_val, channels_test, hold_out_test_channels
from utils import create_data_for_model, create_model, compute_predicted_IOU, evaluate, evaluate_2SEAL

def set_random_seed():
    seed_value = 23
    import numpy as np
    np.random.seed(seed_value)  # NumPy
    import random
    random.seed(seed_value)  # Python
    from tensorflow.random import set_seed
    set_seed(seed_value)  # Tensorflow


def create_config_name(args):
    if args.model_name == "alignment":
        config_name = "alignment"
    else:
        if args.finetune:
            config_name = args.clip_length + " + " + args.model_name + " + " + "finetuned " + args.type_action_emb + " + " + str(args.epochs)
            print("FINETUNING! " + config_name)
        else:
            config_name = args.clip_length + " + " + args.model_name + " + " + args.type_action_emb + " + " + str(args.epochs)

    return config_name


def main():
    set_random_seed()
    args = parse_args()
    config_name = create_config_name(args)

    print("Creating the data features ...")
    train_data, val_data, test_data = \
        create_data_for_model(args.type_action_emb, args.balance,
                            path_all_annotations="data/dict_all_annotations" + args.clip_length + ".json",
                              channels_val=channels_val,
                              channels_test=channels_test,
                              hold_out_test_channels=hold_out_test_channels)


    print("Creating the model ...")
    predicted, list_predictions = create_model(train_data, val_data, test_data, args.epochs, args.balance, config_name)


    print("Evaluating the model ...")
    compute_predicted_IOU(config_name, predicted, test_data, args.clip_length, list_predictions)
    evaluate(config_name, "1p01_5p01")

    evaluate_2SEAL("alignment", "3s + MPU + Bert + 65", "1p01_5p01")

if __name__ == "__main__":
    main()
