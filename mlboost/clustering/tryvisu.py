import sys
from . import visu

# add your data loading function that return data_train and data_test
from .news import load_news
visu.add_loading_dataset_fct('news2', load_news)
visu.main(sys.argv[1:])
