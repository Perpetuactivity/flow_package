import numpy as np


class Const(object):
    __features_labels = [
        "Destination Port",
        "Protocol",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Fwd PSH Flags",
        "Fwd Header Length",
        "Bwd Header Length",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "SYN Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "Down/Up Ratio",
        "Average Packet Size",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
        "Bwd Avg Packets/Bulk",
        "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Packets",
        "Subflow Bwd Bytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ]

    __dtypes = {
        "Destination Port": np.int64,
        "Protocol": np.int64,
        "Flow Duration": np.float32,
        "Total Fwd Packets": np.float32,
        "Total Backward Packets": np.float32,
        "Total Length of Fwd Packets": np.float32,
        "Total Length of Bwd Packets": np.float32,
        "Fwd Packet Length Max": np.float32,
        "Fwd Packet Length Min": np.float32,
        "Fwd Packet Length Mean": np.float32,
        "Fwd Packet Length Std": np.float32,
        "Bwd Packet Length Max": np.float32,
        "Bwd Packet Length Min": np.float32,
        "Bwd Packet Length Mean": np.float32,
        "Bwd Packet Length Std": np.float32,
        "Flow Bytes/s": np.float32,
        "Flow Packets/s": np.float32,
        "Flow IAT Mean": np.float32,
        "Flow IAT Std": np.float32,
        "Flow IAT Max": np.float32,
        "Flow IAT Min": np.float32,
        "Fwd IAT Total": np.float32,
        "Fwd IAT Mean": np.float32,
        "Fwd IAT Std": np.float32,
        "Fwd IAT Max": np.float32,
        "Fwd IAT Min": np.float32,
        "Bwd IAT Total": np.float32,
        "Bwd IAT Mean": np.float32,
        "Bwd IAT Std": np.float32,
        "Bwd IAT Max": np.float32,
        "Bwd IAT Min": np.float32,
        "Fwd PSH Flags": np.float32,
        "Fwd Header Length": np.float32,
        "Bwd Header Length": np.float32,
        "Fwd Packets/s": np.float32,
        "Bwd Packets/s": np.float32,
        "Min Packet Length": np.float32,
        "Max Packet Length": np.float32,
        "Packet Length Mean": np.float32,
        "Packet Length Std": np.float32,
        "Packet Length Variance": np.float32,
        "SYN Flag Count": np.int64,
        "PSH Flag Count": np.int64,
        "ACK Flag Count": np.int64,
        "Down/Up Ratio": np.float32,
        "Average Packet Size": np.float32,
        "Avg Fwd Segment Size": np.float32,
        "Avg Bwd Segment Size": np.float32,
        "Bwd Avg Packets/Bulk": np.float32,
        "Bwd Avg Bulk Rate": np.float32,
        "Subflow Fwd Packets": np.float32,
        "Subflow Fwd Bytes": np.float32,
        "Subflow Bwd Packets": np.float32,
        "Subflow Bwd Bytes": np.float32,
        "Init_Win_bytes_forward": np.float32,
        "Init_Win_bytes_backward": np.float32,
        "act_data_pkt_fwd": np.float32,
        "min_seg_size_forward": np.float32,
        "Active Mean": np.float32,
        "Active Std": np.float32,
        "Active Max": np.float32,
        "Active Min": np.float32,
        "Idle Mean": np.float32,
        "Idle Std": np.float32,
        "Idle Max": np.float32,
        "Idle Min": np.float32,
        "Number Label": np.int64
    }

    @property
    def features_labels(self):
        return self.__features_labels

    @property
    def dtypes(self):
        return self.__dtypes
