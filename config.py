import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default='best_model.pth')

    return parser.parse_args()