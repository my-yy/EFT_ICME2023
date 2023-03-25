from utils import distance_util, path_util
from utils import map_evaluate
from sklearn.metrics import roc_auc_score
import scipy.spatial
import numpy as np
import collections


def calc_ms(all_data, v2emb, f2emb):
    voice1_emb = []
    voice2_emb = []
    face1_emb = []
    face2_emb = []

    for name1, voice1, face1, name2, voice2, face2 in all_data:
        voice1_emb.append(v2emb[voice1])
        voice2_emb.append(v2emb[voice2])
        face1_emb.append(f2emb[face1])
        face2_emb.append(f2emb[face2])

    voice1_emb = np.array(voice1_emb)
    voice2_emb = np.array(voice2_emb)
    face1_emb = np.array(face1_emb)
    face2_emb = np.array(face2_emb)

    dist_vf1 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face1_emb)
    dist_vf2 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face2_emb)
    dist_fv1 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice1_emb)
    dist_fv2 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice2_emb)

    vf_result = dist_vf1 < dist_vf2
    fv_result = dist_fv1 < dist_fv2
    ms_vf = np.mean(vf_result)
    ms_fv = np.mean(fv_result)

    obj = {
        "dist_vf1": dist_vf1,
        "dist_vf2": dist_vf2,
        "dist_fv1": dist_fv1,
        "dist_fv2": dist_fv2,
        "test_data": all_data,  # name1, voice1, face1, name2, voice2, face2
        "result_fv": fv_result,
        "result_vf": vf_result,
        "score_vf": ms_vf,
        "score_fv": ms_fv,
    }
    return ms_vf * 100, ms_fv * 100


def calc_map_value(retrieval_lists, v2emb, f2emb):
    tmp_dic = collections.defaultdict(list)
    for arr in retrieval_lists:
        # 计算每一组的：
        map_vf, map_fv = calc_map_recall_at_k(arr, v2emb, f2emb)
        tmp_dic["map_vf"].append(map_vf)
        tmp_dic["map_fv"].append(map_fv)
    map_fv = np.mean(tmp_dic["map_fv"])
    map_vf = np.mean(tmp_dic["map_vf"])
    return map_vf * 100, map_fv * 100


def handle_1_n(match_list, is_v2f, key2emb):
    tmp_dict = collections.defaultdict(list)
    for voices, faces in match_list:
        if is_v2f:
            prob = voices[0]
            gallery = faces
        else:
            prob = faces[0]
            gallery = voices

        # 1.向量化
        prob_vec = np.array([key2emb[prob]])
        gallery_vec = np.array([key2emb[i] for i in gallery])

        # 2.计算距离
        distances = scipy.spatial.distance.cdist(prob_vec, gallery_vec, 'cosine')
        distances = distances.squeeze()
        assert len(distances) == len(gallery_vec)

        # 3.得到2~N的结果
        for index in range(2, len(gallery) + 1):
            arr = distances[:index]
            is_correct = int(np.argmin(arr) == 0)
            tmp_dict[index].append(is_correct)

    for key, arr in tmp_dict.items():
        tmp_dict[key] = np.mean(arr)
    return tmp_dict


def cosine_similarity(a, b):
    assert len(a.shape) == 2
    assert a.shape == b.shape

    ab = np.sum(a * b, axis=1)
    # (batch_size,)

    a_norm = np.sqrt(np.sum(a * a, axis=1))
    b_norm = np.sqrt(np.sum(b * b, axis=1))
    cosine = ab / (a_norm * b_norm)
    # [-1,1]
    prob = (cosine + 1) / 2.0
    # [0,1]
    return prob


def calc_vrification(the_list, v2emb, f2emb):
    voice_emb = np.array([v2emb[tup[0]] for tup in the_list])
    face_emb = np.array([f2emb[tup[1]] for tup in the_list])
    real_label = np.array([tup[2] for tup in the_list])

    # AUC
    prob = cosine_similarity(voice_emb, face_emb)
    auc = roc_auc_score(real_label, prob)
    return auc * 100


def calc_map_recall_at_k(all_data, v2emb, f2emb):
    # 1.get embedding
    labels = []
    v_emb_list = []
    f_emb_list = []
    for v, f, name in all_data:
        labels.append(name)
        v_emb_list.append(v2emb[v])
        f_emb_list.append(f2emb[f])

    v_emb_list = np.array(v_emb_list)
    f_emb_list = np.array(f_emb_list)

    # 2. calculate distance
    vf_dist = scipy.spatial.distance.cdist(v_emb_list, f_emb_list, 'cosine')
    fv_dist = vf_dist.T

    # 3.map value
    map_vf = map_evaluate.fx_calc_map_label_v2(vf_dist, labels)
    map_fv = map_evaluate.fx_calc_map_label_v2(fv_dist, labels)
    return map_vf, map_fv


def calc_ms_f2v(all_data, v2emb, f2emb):
    voice1_emb = []
    voice2_emb = []
    face1_emb = []

    for face1, voice1, voice2 in all_data:
        voice1_emb.append(v2emb[voice1])
        voice2_emb.append(v2emb[voice2])
        face1_emb.append(f2emb[face1])

    voice1_emb = np.array(voice1_emb)
    voice2_emb = np.array(voice2_emb)
    face1_emb = np.array(face1_emb)

    dist_fv1 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice1_emb)
    dist_fv2 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice2_emb)

    fv_result = dist_fv1 < dist_fv2
    ms_fv = np.mean(fv_result)
    return ms_fv


def calc_ms_v2f(all_data, v2emb, f2emb):
    voice1_emb = []
    face1_emb = []
    face2_emb = []

    for voice1, face1, face2 in all_data:
        voice1_emb.append(v2emb[voice1])
        face1_emb.append(f2emb[face1])
        face2_emb.append(f2emb[face2])

    voice1_emb = np.array(voice1_emb)
    face1_emb = np.array(face1_emb)
    face2_emb = np.array(face2_emb)

    dist_vf1 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face1_emb)
    dist_vf2 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face2_emb)

    vf_result = dist_vf1 < dist_vf2
    ms_vf = np.mean(vf_result)
    return ms_vf
