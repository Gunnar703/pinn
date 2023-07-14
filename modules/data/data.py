import numpy as np
import torch
import os


class Data:
    def __init__(self, data_file="data"):
        files = os.listdir(data_file)
        attr_names = [file.split(".")[0] for file in files]
        data_dict = {
            attr_name: np.loadtxt(os.path.join(data_file, attr_name + ".txt"))
            for attr_name in attr_names
        }

        self.data_dict = data_dict

        self.M = data_dict["M"]
        self.C = data_dict["C"]
        self.K = data_dict["K"]
        self.Y = data_dict["Y"]
        self.t = data_dict["t"]
        self.k_basis = data_dict["k_basis"]
        self.load = data_dict["load"][: len(self.t)]
        self.a0, self.a1 = data_dict["Damp_param"]

        self.torchifiable_params = (
            self.t,
            self.M,
            self.C,
            self.K,
            self.load,
            self.k_basis,
        )

        ## Force dimension hard-coded for now
        self.force_idx = 3
        self.sensor_indices = [1, 3]

        self.vel = np.array(
            [
                data_dict["Vel_3_1_2D"][: len(self.t)],
                data_dict["Vel_3_2D"][: len(self.t)],
                data_dict["Vel_4_1_2D"][: len(self.t)],
                data_dict["Vel_4_2D"][: len(self.t)],
            ]
        )

        self.disp = np.array(
            [
                data_dict["Disp_3_1_2D"][: len(self.t)],
                data_dict["Disp_3_2D"][: len(self.t)],
                data_dict["Disp_4_1_2D"][: len(self.t)],
                data_dict["Disp_4_2D"][: len(self.t)],
            ]
        )

    def torchify(self, device):
        self.device = device
        for param in self.torchifiable_params:
            param = torch.as_tensor(param).to(device)
