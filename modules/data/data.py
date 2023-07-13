import numpy as np
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

        self.vel = np.array(
            [
                data_dict["Vel_3_1_2D"][: len(self.t)],
                data_dict["Vel_3_2D"][: len(self.t)],
                data_dict["Vel_4_1_2D"][: len(self.t)],
                data_dict["Vel_4_2D"][: len(self.t)],
            ]
        )
