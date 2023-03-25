import random
import numpy as np


def random_element(array, need_index=False):
    length = len(array)
    assert length > 0, length
    rand_index = random.randint(0, length - 1)  # 两个边界都能取到
    if need_index:
        return array[rand_index], rand_index
    else:
        return array[rand_index]


def random_elements(array, number):
    return np.random.choice(array, number, replace=False)


def random_elements_sequential(array, number):
    if len(array) == number:
        return array

    assert len(array) > number
    # 剩余长度
    length = len(array) - number + 1
    rand_index = random.randint(0, length - 1)  # 两个边界都能取到
    sub_array = array[rand_index:rand_index + number]
    assert len(sub_array) == number
    return sub_array


def get_selected_indexes(total_frame_count, return_count, ratio):
    # 2）需要先抽取的帧数
    frame_count_need = return_count * ratio
    # 认为仍然是可以选择的：
    assert frame_count_need <= total_frame_count

    # 3)进行抽取
    rand_index_start = np.random.randint(0, total_frame_count - 1 - frame_count_need + 1)
    indexes = []
    for i in range(rand_index_start, rand_index_start + frame_count_need):
        if i % ratio == 0:
            indexes.append(i)
    assert len(indexes) == return_count
    return indexes


if __name__ == "__main__":
    print(get_selected_indexes(9, 3, ratio=2))
    print(get_selected_indexes(9, 3, ratio=2))
    print(get_selected_indexes(9, 3, ratio=2))
    print(get_selected_indexes(9, 3, ratio=2))
    print(get_selected_indexes(9, 3, ratio=2))
    print(get_selected_indexes(9, 3, ratio=2))
