from sklearn.datasets import fetch_20newsgroups

def load_news(fname=None, all_categories=False):
    ''' fname is useless but its there to follow the interface convention'''
    # Load some categories from the training set
    if all_categories:
        categories = None
    else:
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
            ]

    print("Loading 20 newsgroups dataset for categories:")
    print((categories if categories else "all"))
    
    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42)
    
    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42)
    print('data loaded')

    categories = data_train.target_names    # for case categories == None
    
    print(("%d categories" % len(categories)))
    print()

    # split a training set and a test set
    return data_train, data_test, categories
    
