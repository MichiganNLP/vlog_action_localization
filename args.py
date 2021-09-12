import argparse

channels_val = ["2p0", "2p1", "3p0", "3p1"]
channels_test = ["1p0", "1p1", "5p0", "5p1"]
hold_out_test_channels = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--type_action_emb', choices=["ELMo", "Bert"], default="Bert")
    parser.add_argument('-m', '--model_name', choices=["MPU"], default="MPU")
    parser.add_argument('-f', '--finetune', action='store_true')
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('-cl', '--clip_length', choices=["3s"], default="3s")
    parser.add_argument('-b', '--balance', type=bool, choices=[True, False], default=True)
    return parser.parse_args()
