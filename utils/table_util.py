def get_mean_val(value):
    return float(value.split("±")[0])


def trans_obj(obj, base):
    new_obj = {}
    for key, value in obj.items():
        if "test/" not in key:
            new_obj[key] = value
            continue
        value = get_mean_val(value)
        base_value = get_mean_val(base[key])
        diff = base_value - value
        if diff > 0:
            txt = "%.1f↓%.1f" % (value, diff)
        else:
            txt = "%.1f↑%.1f" % (value, -diff)
        new_obj[key] = txt
    return new_obj


def to_table1(objlist, base):
    show_list = [trans_obj(obj, base) for obj in objlist]
    keys = ['name', 'test/auc', 'test/auc_g', 'test/ms_v2f', 'test/ms_f2v', 'test/map_v2f', 'test/map_f2v']
    for obj in show_list:
        values = [obj[k] for k in keys]
        txt = "\t".join(values)
        print(txt)
