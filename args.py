import argparse

channels_val = ["2p0", "2p1", "3p0", "3p1"]
channels_test = ["1p0", "1p1", "5p0", "5p1"]
# channels_test = ["1p1"]
# hold_out_test_channels = ["9p0", "9p1", "10p0", "10p1"]
# hold_out_test_channels = ["9p0", "9p1"]
# hold_out_test_channels = ["9p0", "9p1", "10p0", "10p1","8p0","8p1","6p0","6p1","7p0","7p1"]
hold_out_test_channels = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--type_action_emb', choices=["GloVe", "ELMo", "Bert", "DNT"], default="Bert")
    parser.add_argument('-m', '--model_name', choices=["MPU", "cosine sim", "alignment", "system max", "main", "Main"], default="MPU")
    parser.add_argument('-f', '--finetune', action='store_true')
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('-cl', '--clip_length', choices=["3s", "10s"], default="3s")
    parser.add_argument('-b', '--balance', type=bool, choices=[True, False], default=True)
    parser.add_argument('-c', '--add_cluster', action='store_true')
    parser.add_argument('-ol', '--add_obj_label', choices=["original", "hands", "none"], default="none")
    parser.add_argument('-of', '--add_obj_feat', choices=["original", "hands", "none"], default="none")
    return parser.parse_args()
