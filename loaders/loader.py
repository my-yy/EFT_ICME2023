import collections
import numpy as np
import torch
from utils import pickle_util, sample_util
from torch.utils.data import DataLoader


class DataSet(torch.utils.data.Dataset):

    def __init__(self, features, full_length, big_batch, batch_size, batch_construction_mode):
        self.train_names = pickle_util.read_pickle("./dataset/info/train_valid_test_names.pkl")["train"]
        self.name2tracks = pickle_util.read_pickle("./dataset/info/name2tracks.pkl")
        self.batch_size = batch_size
        self.batch_construction_mode = batch_construction_mode
        if batch_construction_mode.startswith("sbc_"):
            self.std_ratio = float(batch_construction_mode.split("_")[-1])
            print("std_ratio", self.std_ratio)

        all_tracks = []
        for name in self.train_names:
            all_tracks += self.name2tracks[name]
        # This track has corrupted, thus we remove it:
        all_tracks = [i for i in all_tracks if i != "Moon_Bloodgood/1.6/JBhcgwl2pO0/2"]

        self.all_tracks = all_tracks

        self.full_length = full_length

        self.features = features
        self.big_batch_size = big_batch

    def __len__(self):
        return self.full_length

    def load_big_batch(self):
        big_tracks = sample_util.random_elements(self.all_tracks, self.big_batch_size)
        face_list = []
        voice_list = []
        track2array = {}
        for track in big_tracks:
            array = self.load_one(track)
            track2array[track] = array
            face = array[0]
            voice = array[-2]
            face_list.append(face)
            voice_list.append(voice)
        return big_tracks, face_list, voice_list, track2array

    def track_name_list2final_batch(self, need_tracks, track2array):
        result = collections.defaultdict(list)
        assert len(need_tracks) == self.batch_size
        for track in need_tracks:
            array = track2array[track]
            for i in range(len(self.features)):
                result[i].append(array[i])
        data = [np.array(result[i]) for i in range(len(self.features))]
        data = [torch.FloatTensor(d) for d in data]
        return data

    def calc_repeat_rate(self, tracks):
        names = set([t.split("/")[0] for t in tracks])
        return 1 - len(names) / len(tracks)

    def do_filter_by_sbc(self, tracks, face_list, voice_list):
        face_batch = np.array(face_list)
        voice_batch = np.array(voice_list)
        face_sim = face_batch @ face_batch.T
        voice_sim = voice_batch @ voice_batch.T
        sim_cated = np.array([face_sim, voice_sim])
        sim_matrix = np.min(sim_cated, axis=0)

        np.fill_diagonal(sim_matrix, 0)
        stat_min = sim_matrix.mean()
        stat_std = sim_matrix.std()
        threshold = stat_min + self.std_ratio * stat_std

        filtered_tracks = [tracks[0]]
        has_names = set()
        has_names.add(tracks[0].split("/")[0])

        for i in range(1, len(tracks)):
            track = tracks[i]
            cur_name = track.split("/")[0]
            if cur_name in has_names:
                sub_array = sim_matrix[i][0:i]
                sub_array_max = sub_array.max()
                if sub_array_max <= threshold:
                    filtered_tracks.append(track)
                    has_names.add(cur_name)
                else:
                    sim_matrix[:, i] = 0
            else:
                filtered_tracks.append(track)
                has_names.add(cur_name)
            if len(filtered_tracks) == self.batch_size:
                break
        return filtered_tracks

    def do_filter_by_real_labels(self, tracks):
        nameset = set()
        filtered_tracks = []
        for t in tracks:
            name = t.split("/")[0]
            if name in nameset:
                continue
            nameset.add(name)
            filtered_tracks.append(t)
            if len(filtered_tracks) == self.batch_size:
                break
        return filtered_tracks

    def __getitem__(self, index):
        big_tracks, face_list, voice_list, track2array = self.load_big_batch()

        if self.batch_construction_mode == "vanilla":
            tracks = big_tracks[0:self.batch_size]

        elif self.batch_construction_mode == "label":
            tracks = self.do_filter_by_real_labels(big_tracks)

        elif self.batch_construction_mode.startswith("sbc"):
            tracks = self.do_filter_by_sbc(big_tracks, face_list, voice_list)
        else:
            raise Exception("error batch method")

        data = self.track_name_list2final_batch(tracks, track2array)

        rate_before = self.calc_repeat_rate(big_tracks)
        rate_after = self.calc_repeat_rate(tracks)
        the_index = big_tracks.tolist().index(tracks[-1])
        info = {
            "loader/before": rate_before,
            "loader/after": rate_after,
            "loader/desc": rate_before - rate_after,
            "loader/endIndex": the_index,
        }
        return data, info

    def load_one(self, track):
        track = "./dataset/features/" + track.replace("/1.6/", "/")
        tmp_dict = pickle_util.read_pickle(track + "/compact.pkl")
        # e.g.: ./dataset/features/Kimberly_Williams-Paisley/UPWCLTOQARI/13/compact.pkl'
        all_features = []
        for f in self.features:
            data = tmp_dict[f]
            all_features.append(data)
        return all_features
