from abc import ABC

import torch.utils.data

from lichee import plugin
from lichee.utils.tfrecord.reader import read_single_record_with_spec_index
from lichee.utils.tfrecord.reader import read_org_single_record_with_spec_index
from .dataset_base import BaseDataset


@plugin.register_plugin(plugin.PluginType.DATA_LOADER, "dataset_mem")
class DatasetMem(torch.utils.data.Dataset, BaseDataset, ABC):
    def __init__(self, cfg, data_file, desc_file, training=True):
        super().__init__(cfg, data_file, desc_file, training)

    def __getitem__(self, index):
        """
        get transformed data with index

        :param index: data index
        :return: transformed data
        """
        data_file_index, (start_offset, end_offset) = self.get_nth_data_file(index)
        tfrecord_data_file = self.tfrecord_data_file_list[data_file_index]
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset, self.description)
        return self.transform(row)


@plugin.register_plugin(plugin.PluginType.DATA_LOADER, "dataset_org_mem")
class DatasetOrgMem(torch.utils.data.Dataset, BaseDataset, ABC):
    def __init__(self, cfg, data_file, desc_file, training=True):
        super().__init__(cfg, data_file, desc_file, training)

    def __getitem__(self, index):
        """
        get transformed data with index

        :param index: data index
        :return: transformed data
        """
        data_file_index, (start_offset, end_offset) = self.get_nth_data_file(index)
        tfrecord_data_file = self.tfrecord_data_file_list[data_file_index]
        row = read_org_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset, self.description)
        return row

    def collate(self, batch):
        """
        collate data in a batch

        :param batch: list of item feature map [item1, item2, item3 ...], item with {key1:feature1, key2:feature2}
        :return: batched feature map for model with format {key1: batch_feature1, key2: batch_feature2...}
        """
        # record = {}
        #
        # for parser in self.parsers:
        #     collate_result = parser.collate(batch)
        #     if collate_result is not None:
        #         record.update(collate_result)

        return batch
