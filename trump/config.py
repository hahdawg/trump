import os

join = os.path.join

here = os.path.dirname(os.path.abspath(__file__))
data_dir = join(here, "src_data")
data_path = join(data_dir, "tweets.csv")
log_dir = join(here, "log")
model_dir = join(here, "model")
