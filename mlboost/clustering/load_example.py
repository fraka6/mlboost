class Data:pass

def load_default(fname, ratio=.7, sep='\t'):
    ''' formats: default = features, target '''
    data = []
    target = []
    reader = open(fname, 'r')
    header = reader.readline()
    for line in reader:
        features = line.strip().split(sep)
        data.append(features[:-1])
        target.append(features[-1])
    
    train = Data()
    test = Data()
    
    idx = ratio*len(data)
    train.data = data[0:idx]
    test.data = data[idx:]
    train.target = target[0:idx]
    test.target = target[idx:]
    
    return train, test

    
    
