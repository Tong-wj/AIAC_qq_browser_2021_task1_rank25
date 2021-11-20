import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir, label_file):
        # data_dir = "./data/pairwise_self_single_half"
        # label_file = "./data/pairwise/label_half.tsv"
        self.sim_score = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rk1, rk2, score = line.split('\t')
                self.sim_score.append(float(score))
        self.lsh_data_dir = data_dir + "/lsh"
        self.rsh_data_dir = data_dir + "/rsh"

    def __len__(self):
        return len(self.sim_score)

    def __getitem__(self, index):
        lsh_data_path = self.lsh_data_dir + "/" + str(index) + ".pt"
        rsh_data_path = self.rsh_data_dir + "/" + str(index) + ".pt"
        lsh_data = torch.load(lsh_data_path)
        rsh_data = torch.load(rsh_data_path)
        score = self.sim_score[index]

        return lsh_data, rsh_data, score
