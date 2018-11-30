
'''
def get_data(fil, index):
    with open(fil, "r") as f:
        raw = f.read().split("\r\n")
    print(raw)
    data_raw = [[eval(y) for y in x.split(' ')] for x in raw[:-1]]
    print(data_raw)
    numx = data_raw[0][0]
    print(numx)
    y_ = [x[index+numx] for x in data_raw[1:]]
    print(y_)
    yd = max(y_) - min(y_)
    ymin = min(y_)
    dataset = [[tuple(x[0:numx]), tuple([int(i==(x[index+numx]-ymin)) for i in range(yd+1)])] for x in data_raw[1:]]
    return dataset

print(get_data('test.zfy', 0))
'''

def get_data(fil, index):
    with open(fil, "r") as f:
        raw = f.read().split("\r\n")
    print(raw)
    print('----------------------------------------\n\n\n\n----------------------------------------')
    data_raw = [[eval(y) for y in x.split(' ')] for x in raw[:-1]]
    print(data_raw)
    print('----------------------------------------\n\n\n\n----------------------------------------')
    numx = data_raw[0][0]
    print(numx)
    y_ = [x[index+numx] for x in data_raw[1:]]
    print(y_)
    yd = max(y_) - min(y_)
    ymin = min(y_)
    print(max(y_))
    print(min(y_))
    print('----------------------------------------\n\n\n\n----------------------------------------')
    dataset = [[tuple(x[0:numx]), tuple([int(i==(x[index+numx]-ymin)) for i in range(yd+1)])] for x in data_raw[1:]]
    print(dataset)
    return dataset

get_data('testdata.in', 0)
