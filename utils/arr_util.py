import copy


def expand_arr(arr, aim_length):
    arr = copy.deepcopy(arr)
    sup_len = aim_length - len(arr)
    if sup_len > 0:
        for i in range(sup_len):
            arr.append(arr[-1])
    assert len(arr) >= aim_length
    return arr


def expand_arr_v2(arr, aim_length):
    if len(arr) >= aim_length:
        return arr

    new_arr = copy.deepcopy(arr)
    while True:
        new_arr += arr
        if len(new_arr) >= aim_length:
            return new_arr
