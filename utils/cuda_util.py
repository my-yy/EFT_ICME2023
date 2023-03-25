
def to_cuda(data):
    result = []
    for item in data:
        if type(item) == list:
            item = [i.cuda() for i in item]
        else:
            item = item.cuda()
        result.append(item)
    return result
