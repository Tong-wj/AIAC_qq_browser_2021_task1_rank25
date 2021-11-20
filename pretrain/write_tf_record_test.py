import json

import torch
import numpy as np
from lichee.utils.tfrecord import example_pb2
from lichee.utils.tfrecord.writer import TFRecordWriter

data_path = 'data/pairwise_self_single/lsh/0.pt'
batch = torch.load(data_path)
print(batch)

# example = {
#
#     "frame_feature": (batch['frame_feature'].numpy().tobytes(), 'byte'),
#
#     "title": (batch['title'].numpy().tobytes(), 'byte'),
#
#     "tag_id": (batch['tag_id'].numpy().tobytes(), 'byte'),
#
#     "similarity": (1.0, 'float'),
# }

# example = {
#
#     "frame_feature": (batch['frame_feature'], 'byte'),
#
#     "title": (batch['title'], 'byte'),
#
#     "tag_id": (batch['tag_id'], 'byte'),
#
#     "similarity": (1.0, 'float'),
# }


example_1 = {

    "frame_feature": (batch['frame_feature'].numpy().tobytes(), 'bytes'),

    "title": ("中秋节快乐！".encode(encoding="utf-8"), 'byte'),

    "id": (batch['id'].encode(), 'byte'),

    "tag_id": (batch['tag_id'].numpy(), 'int'),

    "similarity": (1.0, 'float'),
}

example_2 = {

    "frame_feature": (batch['frame_feature'].numpy().tobytes(), 'bytes'),

    "title": ("国庆节快乐！".encode(encoding="utf-8"), 'byte'),

    "id": (batch['id'].encode(), 'byte'),

    "tag_id": (batch['tag_id'].numpy(), 'int'),

    "similarity": (0.5, 'float'),
}


print(example_1)

save_path = 'save_data.tfrecords'
writer = TFRecordWriter(save_path)
writer.write(example_1)
writer.write(example_2)
writer.close()
