import json
import logging
import os
import io

import numpy as np
import pandas as pd
import scipy
import torch
import torch.multiprocessing
import tqdm

from lichee import plugin
from lichee.core.trainer.trainer_base import TrainerBase
from lichee.utils import storage
from lichee.utils import sys_tmpfile
from lichee.utils.convertor import torch_nn_convertor
from .my_dataset import MyDataset
from lichee.utils.tfrecord.writer import TFRecordWriter


def float_to_str(float_list):
    return ','.join(['%f' % val for val in float_list])


@plugin.register_plugin(plugin.PluginType.TASK, 'mlm_cls')
class MlmCls(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    predicting tag_id
    """

    def __init__(self, cfg):
        """
        :param cfg: concat_cls config defined in your_config.yaml
        """
        super().__init__()
        self.cfg = cfg
        self.linear_tag = torch.nn.Linear(cfg['VIDEO_SIZE'], cfg['NUM_CLASSES_TAG'])
        self.linear_mlm = torch.nn.Linear(cfg['TITLE_SIZE'], cfg['NUM_CLASSES_VOCAB'])
        self.fc_hidden = torch.nn.Linear(cfg['INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.loss_func = None
        self.init_loss()

    def forward(self, video_feature, title_feature, label=None):
        """
        :param video_feature: video feature extracted from frame representation
        :param title_feature: title feature extracted from title representation
        :param label: masked label
        :return: (predictions, embeddings), model loss
        """

        pred_tag = self.linear_tag(video_feature)
        loss_tag = None
        if label is not None:
            label_tag = label[self.cfg["LABEL_TAG_KEY"]].float()
            loss_tag = self.loss_func_bce(pred_tag, label_tag)
            if 'SCALE' in self.cfg['LOSS_BCE']:
                loss_tag = loss_tag * self.cfg['LOSS_BCE']['SCALE']
        pred_tag = torch.sigmoid(pred_tag)

        pred_mlm = self.linear_mlm(title_feature)
        loss_mlm = torch.tensor(0).float().to('cuda')
        if label is not None:
            label_mlm = label[self.cfg["LABEL_MLM_KEY"]]
            dim_label_mlm = label_mlm.shape
            for i in range(dim_label_mlm[0]):
                for j in range(dim_label_mlm[1]):
                    if label_mlm[i, j] != 0:
                        loss_mlm += self.loss_func_ce(pred_mlm[i:i + 1, j, :], label_mlm[i:i + 1, j])
            if 'SCALE' in self.cfg['LOSS_CE']:
                loss_mlm = loss_mlm * self.cfg['LOSS_CE']['SCALE']

        feature = torch.cat([video_feature, title_feature[:, 0]], dim=1)
        embedding = self.fc_hidden(feature)
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return (pred_tag, normed_embedding), loss_tag, loss_mlm


    def init_loss(self):
        loss_bce = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg['LOSS_BCE']['NAME'])
        self.loss_func_bce = loss_bce.build(self.cfg['LOSS_BCE'])
        loss_ce = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg['LOSS_CE']['NAME'])
        self.loss_func_ce = loss_ce.build(self.cfg['LOSS_CE'])


@plugin.register_plugin(plugin.PluginType.TASK, 'concat_cls')
class ConcatCls(torch.nn.Module):
    def __init__(self, cfg):
        """
        :param cfg: concat_cls config defined in your_config.yaml
        """
        super().__init__()
        self.cfg = cfg
        self.drop_rate = 0.5
        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.fc_hidden = torch.nn.Linear(cfg['INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.fc_logits = torch.nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
        self.loss_func = None
        self.init_loss()

    def forward(self, video_feature, title_feature, label=None):
        """
        :param video_feature: video feature extracted from frame representation
        :param title_feature: title feature extracted from title representation
        :param label: classification target
        :return: (predictions, embeddings), model loss
        """
        # title_feature_0 = title_feature[:, 0]
        # title_feature_1 = title_feature[:, 1]

        title_feature = title_feature[:, 0]
        # title_feature = torch.mean(title_feature, dim=1)

        feature = torch.cat([video_feature, title_feature], dim=1)
        # feature = title_feature
        # feature = self.dropout(feature)
        embedding = self.fc_hidden(feature)
        # feature = video_feature
        # embedding = feature
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        pred = self.fc_logits(torch.relu(embedding))
        loss = None
        if label is not None:
            label = label[self.cfg["LABEL_KEY"]].float()
            loss = self.loss_func(pred, label)
            if 'SCALE' in self.cfg['LOSS']:
                loss = loss * self.cfg['LOSS']['SCALE']
        pred = torch.sigmoid(pred)
        return (pred, normed_embedding), loss

    def init_loss(self):
        loss = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg['LOSS']['NAME'])
        self.loss_func = loss.build(self.cfg['LOSS'])


@plugin.register_plugin(plugin.PluginType.TRAINER, 'embedding_trainer')
class EmbeddingTrainer(TrainerBase):

    def __init__(self, config, init_model=True):
        """
        :param config: the global trainer config, defined by your_config.yaml
        :param init_model: whether initialize the model or not
        """
        super().__init__(config, init_model)

    def report_step(self, step):
        metric = self.metrics[0]
        metric_info = metric.calc()
        metric_info['loss'] = self.loss
        metric_info['step'] = step
        logging.info(
            "Step {step}, precision: {precision:.4f}, recall: {recall:.4f}, loss: {loss:.4f}".format_map(metric_info))

    def report_eval_step(self, metric_info):
        print(metric_info)
        print(type(metric_info))
        logging.info(
            "EVAL EPOCH {epoch}, precision: {precision:.4f}, recall: {recall:.4f}, loss: {loss:.4f}".format_map(
                metric_info))
        self.temporary_map.update(metric_info)

    def pairwise_data_prepare(self, checkpoint_file='', dataset_key="DATA_PREPARE"):
        """
        :param checkpoint_file: the checkpoint used to evaluate the dataset
        :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
        :return:
        """

        # torch.multiprocessing.set_sharing_strategy('file_system')
        if checkpoint_file:
            self.load_checkpoint_for_eval(checkpoint_file)
        self.model.train()
        dataset_config = self.cfg.DATASET[dataset_key]
        dataset_loader = self.gen_dataloader(dataset_config, training=False)
        self.save_config_file()
        dataset_list = []
        id_list = []
        # 先将所有的视频数据读入，记录对应的id
        for _, batch in tqdm.tqdm(enumerate(dataset_loader)):
            row = batch[0]
            # print(row)
            id = bytes(row['id']).decode('utf-8')
            # print(id)
            dataset_list.append(row)
            id_list.append(id)
            # print(id_list)

        # # 二维列表转一维，用sum方式
        # id_list = sum(id_list, [])

        # 读入 pairwise_label, 根据id号，每次选择256个 pairwise 输入网络
        lsh_id = []
        rsh_id = []
        sim_score = []
        label_file = storage.get_storage_file(dataset_config['LABEL_FILE'])
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rk1, rk2, score = line.split('\t')
                lsh_id.append(rk1)
                rsh_id.append(rk2)
                sim_score.append(float(score))

        # 考虑之后设置到配置文件中
        batch_size = 1
        step_number = len(lsh_id) // batch_size

        # 不重复id的放到训练集，重复id的放到验证集
        id_pairwise_list = []

        # train_save_path = 'data/pairwise_self_tfrecord_balance/pairwise_train.tfrecords'
        # val_save_path = 'data/pairwise_self_tfrecord_balance/pairwise_val_with_cut.tfrecords'
        # train_writer = TFRecordWriter(train_save_path)
        # val_writer = TFRecordWriter(val_save_path)
        #
        # # for step in range(0, 67899):
        # for step in range(0, 52207):
        #     print(f"-----------------------Saving data step {step}/{step_number - 1}-----------------------")
        #     lsh_id_batch = lsh_id[step]
        #     rsh_id_batch = rsh_id[step]
        #     for i, id_list_tmp in enumerate(id_list):
        #         if id_list_tmp == lsh_id_batch:
        #             lsh_id_index = i
        #             break
        #     for i, id_list_tmp in enumerate(id_list):
        #         if id_list_tmp == rsh_id_batch:
        #             rsh_id_index = i
        #             break
        #     # print("lsh_id_index length:", len(lsh_id_index))
        #     # print("rsh_id_index length:", len(rsh_id_index))
        #     #
        #     lsh_data_batch = dataset_list[lsh_id_index]
        #     rsh_data_batch = dataset_list[rsh_id_index]
        #
        #     example = {
        #         "lsh_id": (lsh_data_batch['id'], 'byte'),
        #         "lsh_tag_id": (lsh_data_batch['tag_id'], 'int'),
        #         "lsh_category_id": (lsh_data_batch['category_id'], 'int'),
        #         "lsh_title": (lsh_data_batch['title'], 'byte'),
        #         "lsh_frame_feature": (lsh_data_batch['frame_feature'], 'bytes'),
        #
        #         "rsh_id": (rsh_data_batch['id'], 'byte'),
        #         "rsh_tag_id": (rsh_data_batch['tag_id'], 'int'),
        #         "rsh_category_id": (rsh_data_batch['category_id'], 'int'),
        #         "rsh_title": (rsh_data_batch['title'], 'byte'),
        #         "rsh_frame_feature": (rsh_data_batch['frame_feature'], 'bytes'),
        #
        #         "similarity": (sim_score[step], 'float'),
        #     }
        #
        #     if (lsh_id_batch in id_pairwise_list) or (rsh_id_batch in id_pairwise_list):
        #         # 验证集
        #         val_writer.write(example)
        #     else:
        #         # 训练集
        #         id_pairwise_list.append(lsh_id_batch)
        #         id_pairwise_list.append(rsh_id_batch)
        #         train_writer.write(example)
        # train_writer.close()
        # val_writer.close()
        # print("数据（划分训练集和验证集）准备完毕！")


        # lsh_id = []
        # rsh_id = []
        # sim_score = []
        # label_file = storage.get_storage_file(dataset_config['LABEL_FILE_CUT'])
        # with open(label_file, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip()
        #         rk1, rk2, score = line.split('\t')
        #         lsh_id.append(rk1)
        #         rsh_id.append(rk2)
        #         sim_score.append(float(score))
        #
        # # 考虑之后设置到配置文件中
        # batch_size = 1
        # step_number = len(lsh_id) // batch_size
        #
        # for step in range(0, 15692):
        #     print(f"-----------------------Saving data step {step}/{step_number - 1}-----------------------")
        #     lsh_id_batch = lsh_id[step]
        #     rsh_id_batch = rsh_id[step]
        #     for i, id_list_tmp in enumerate(id_list):
        #         if id_list_tmp == lsh_id_batch:
        #             lsh_id_index = i
        #             break
        #     for i, id_list_tmp in enumerate(id_list):
        #         if id_list_tmp == rsh_id_batch:
        #             rsh_id_index = i
        #             break
        #     # print("lsh_id_index length:", len(lsh_id_index))
        #     # print("rsh_id_index length:", len(rsh_id_index))
        #     #
        #     lsh_data_batch = dataset_list[lsh_id_index]
        #     rsh_data_batch = dataset_list[rsh_id_index]
        #
        #     example = {
        #         "lsh_id": (lsh_data_batch['id'], 'byte'),
        #         "lsh_tag_id": (lsh_data_batch['tag_id'], 'int'),
        #         "lsh_category_id": (lsh_data_batch['category_id'], 'int'),
        #         "lsh_title": (lsh_data_batch['title'], 'byte'),
        #         "lsh_frame_feature": (lsh_data_batch['frame_feature'], 'bytes'),
        #
        #         "rsh_id": (rsh_data_batch['id'], 'byte'),
        #         "rsh_tag_id": (rsh_data_batch['tag_id'], 'int'),
        #         "rsh_category_id": (rsh_data_batch['category_id'], 'int'),
        #         "rsh_title": (rsh_data_batch['title'], 'byte'),
        #         "rsh_frame_feature": (rsh_data_batch['frame_feature'], 'bytes'),
        #
        #         "similarity": (sim_score[step], 'float'),
        #     }
        #
        #     val_writer.write(example)
        #


        # 不划分训练集和验证集
        save_path = 'data/pairwise_self_tfrecord_balance/pairwise.tfrecords'
        writer = TFRecordWriter(save_path)

        # for step in range(0, 67899):
        for step in range(0, 58899):
            print(f"-----------------------Saving data step {step}/{step_number - 1}-----------------------")
            lsh_id_batch = lsh_id[step]
            rsh_id_batch = rsh_id[step]
            for i, id_list_tmp in enumerate(id_list):
                if id_list_tmp == lsh_id_batch:
                    lsh_id_index = i
                    break
            for i, id_list_tmp in enumerate(id_list):
                if id_list_tmp == rsh_id_batch:
                    rsh_id_index = i
                    break
            # print("lsh_id_index length:", len(lsh_id_index))
            # print("rsh_id_index length:", len(rsh_id_index))
            #
            lsh_data_batch = dataset_list[lsh_id_index]
            rsh_data_batch = dataset_list[rsh_id_index]

            example = {
                "lsh_id": (lsh_data_batch['id'], 'byte'),
                "lsh_tag_id": (lsh_data_batch['tag_id'], 'int'),
                "lsh_category_id": (lsh_data_batch['category_id'], 'int'),
                "lsh_title": (lsh_data_batch['title'], 'byte'),
                "lsh_frame_feature": (lsh_data_batch['frame_feature'], 'bytes'),

                "rsh_id": (rsh_data_batch['id'], 'byte'),
                "rsh_tag_id": (rsh_data_batch['tag_id'], 'int'),
                "rsh_category_id": (rsh_data_batch['category_id'], 'int'),
                "rsh_title": (rsh_data_batch['title'], 'byte'),
                "rsh_frame_feature": (rsh_data_batch['frame_feature'], 'bytes'),

                "similarity": (sim_score[step], 'float'),
            }
            writer.write(example)
        writer.close()

        print("数据（不划分训练集和测试集）准备完毕！")

    def finetune_with_spearman(self, checkpoint_file='', dataset_key="FINETUNE"):
        """
        :param checkpoint_file: the checkpoint used to evaluate the dataset
        :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
        :return:
        """
        # dataset_config = self.cfg.DATASET[dataset_key]
        # self.save_config_file()

        if checkpoint_file:
            logging.info('Loading pretrained model...')
            self.load_checkpoint_for_finetune(checkpoint_file)
            logging.info('Finished!')
        # print(self.model)
        self.model.train()

        # for p in self.model.module.module.representation_model_arr[1].parameters():
        #     # print(p)
        #     p.requires_grad = False

        self.optimizer.zero_grad()

        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.1)

        print("计算相似度损失...")

        train_dataset_config = self.cfg.DATASET['FINETUNE_TRAIN']
        train_dataset_loader = self.gen_dataloader(train_dataset_config, training=True)
        val_dataset_config = self.cfg.DATASET['FINETUNE_VAL']
        val_dataset_loader = self.gen_dataloader(val_dataset_config, training=False)
        negative_dataset_config = self.cfg.DATASET['FINETUNE_TRAIN']
        negative_dataset_loader = self.gen_dataloader(negative_dataset_config, training=True)
        self.save_config_file()

        # self.model_config_file = 'embedding_split_train.yaml'
        # self.init_config()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_func = torch.nn.MSELoss(reduction='sum')
        loss_func_l1 = torch.nn.L1Loss(reduction='sum')
        fc = torch.nn.Linear(1, 21)
        fc.to(device)
        loss_func_ce = torch.nn.CrossEntropyLoss(reduction='sum')

        constant = 0.05 * torch.randint(1, 9, (1,))

        for epoch in range(1, self.cfg.TRAINING.EPOCHS + 1):
            train_sim_score_predict_list = []
            train_sim_score_label_list = []
            # negative_dataset_config = self.cfg.DATASET['FINETUNE_TRAIN']
            # negative_dataset_loader = self.gen_dataloader(negative_dataset_config, training=True)



            for step, batch in enumerate(train_dataset_loader):
                # negative_batch = next(iter(negative_dataset_loader))

                self.optimizer.zero_grad()
                self.model.train()
                lsh_batch = {'lsh_frame_feature': batch['lsh_frame_feature'], 'tag_id':batch['lsh_tag_id'], 'id': batch['lsh_id'],
                             'lsh_title': batch['lsh_title'], 'org_title': batch['org_lsh_title']}
                # 之所以字典的key都用lsh开头，是因为模型topo中设置的输入就是以lsh开头的
                rsh_batch = {'lsh_frame_feature': batch['rsh_frame_feature'], 'tag_id': batch['rsh_tag_id'], 'id': batch['rsh_id'],
                             'lsh_title': batch['rsh_title'], 'org_title': batch['org_rsh_title']}
                # negative_lsh_batch = {'lsh_frame_feature': negative_batch['lsh_frame_feature'], 'tag_id':negative_batch['lsh_tag_id'], 'id': negative_batch['lsh_id'],
                #              'lsh_title': negative_batch['lsh_title'], 'org_title': negative_batch['org_lsh_title']}

                sim_score = torch.Tensor(batch['similarity'])
                lsh_inputs = self.get_inputs_batch(lsh_batch)
                rsh_inputs = self.get_inputs_batch(rsh_batch)
                # negative_lsh_inputs = self.get_inputs_batch(negative_lsh_batch)

                lsh_label_keys, lsh_labels = self.get_label_batch(lsh_batch)
                rsh_label_keys, rsh_labels = self.get_label_batch(rsh_batch)
                # negative_lsh_label_keys, negative_lsh_labels = self.get_label_batch(negative_lsh_batch)

                lsh_inputs[lsh_label_keys[0]] = lsh_labels[0]
                rsh_inputs[rsh_label_keys[0]] = rsh_labels[0]
                # negative_lsh_inputs[negative_lsh_label_keys[0]] = negative_lsh_labels[0]
                # lsh_inputs = lsh_inputs.squeeze(dim=0)
                # rsh_inputs = rsh_inputs.squeeze(dim=0)

                (_, lsh_embedding), loss_1 = self.model(lsh_inputs)
                (_, rsh_embedding), loss_2 = self.model(rsh_inputs)
                # (_, negative_lsh_embedding), _ = self.model(negative_lsh_inputs)
                sim_score_predict = torch.sum(lsh_embedding * rsh_embedding, dim=1)
                # negative_sim_score_predict = torch.sum(lsh_embedding * negative_lsh_embedding, dim=1)
                # sim_score = sim_score + 0.05 * torch.rand_like(sim_score)
                # sim_score = sim_score + 0.05 * (torch.rand_like(sim_score) - 0.5)
                # sim_score = sim_score + 0.05
                # sim_score = sim_score + 0.1
                # sim_score = sim_score + 0.15
                # sim_score = sim_score + 0.2
                # sim_score = sim_score + 0.25
                # sim_score = sim_score + 0.30
                # sim_score = sim_score + 0.35
                # sim_score = sim_score + 0.4
                sim_score = sim_score + 0.25 * (1-sim_score)
                # sim_score = sim_score + 0.27
                # sim_score = sim_score + 0.23
                # sim_score = sim_score + 0.35 * (1-sim_score)
                # sim_score = sim_score + 0.25 * torch.rand_like(sim_score)
                # sim_score += constant
                # sim_score = 2*(sim_score+0.5)
                # sim_score = torch.clamp(sim_score, 0.0, 1.0)

                sim_score = sim_score.to(device)
                # sim_score_predict += 0.05 * (torch.rand_like(sim_score_predict) - 0.5)

                # with torch.no_grad():
                #     residual = sim_score_predict - torch.round(sim_score_predict * 20) / 20

                # sim_score_predict = sim_score_predict - residual
                sim_score_predict = sim_score_predict


                loss_3 = loss_func(sim_score_predict, sim_score)
                # loss_3 = loss_func_l1(sim_score_predict, sim_score)
                sim_score_predict = sim_score_predict.detach().cpu().numpy()
                sim_score = sim_score.detach().cpu().numpy()
                train_sim_score_predict_list.extend(sim_score_predict.astype(np.float16).tolist())
                train_sim_score_label_list.extend(sim_score.astype(np.float16).tolist())
                # loss = loss_1 + loss_2 + loss_3 + loss_4
                loss = loss_1 + loss_2 + loss_3
                # loss = loss_1 + loss_3
                # loss = loss_3


                # sim_score = torch.clamp(sim_score, 0.0, 1.0)
                # sim_score = sim_score * 20
                # # print(sim_score)
                # sim_score = sim_score.to(device)
                #
                # sim_score_predict = fc(sim_score_predict.unsqueeze(dim=1))
                #
                # # loss_3 = loss_func(sim_score_predict, sim_score)
                #
                # loss_3 = loss_func_ce(sim_score_predict, sim_score.long())
                #
                # # loss_4 = negative_sim_score_predict.sum().pow(2)
                #
                # # sim_score_predict = sim_score_predict.detach().cpu().numpy()
                # # sim_score = sim_score.detach().cpu().numpy()
                # # train_sim_score_predict_list.extend(sim_score_predict.astype(np.float16).tolist())
                # # train_sim_score_label_list.extend(sim_score.astype(np.float16).tolist())
                #
                # # loss = loss_1 + loss_2 + loss_3 + loss_4
                # loss = loss_1 + loss_2 + loss_3
                # # loss = loss_1 + loss_3
                # # loss = loss_3



                if self.cfg.RUNTIME.AUTOCAST:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if self.cfg.RUNTIME.AUTOCAST:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                if (step + 1) % self.cfg.RUNTIME.REPORT_STEPS == 0:
                    logging.info('step: {}/{}   loss_1: {}  loss_2: {}  loss_3: {}   loss: {}'
                                 .format(step, len(train_dataset_loader), loss_1, loss_2, loss_3, loss))
                    # logging.info('step: {}/{}   constant: {}   loss_1: {}  loss_2: {}  loss_3: {}   loss: {}'
                    #              .format(step, len(train_dataset_loader), constant, loss_1, loss_2, loss_3, loss))
                    # constant = 0.05 * torch.randint(1, 9, (1,))
                    # print(constant)

                # logging.info(
                #     'step: {}/{}   loss: {}'.format(step, len(train_dataloader), loss))

                # if step % 10 == 0:
                #     self.model.eval()
                #     sim_score_predict_list = []
                #     sim_score_label_list = []
                #     for step, batch in tqdm.tqdm(enumerate(val_dataset_loader)):
                #         lsh_batch = {'lsh_frame_feature': batch['lsh_frame_feature'], 'tag_id': batch['lsh_tag_id'],
                #                      'id': batch['lsh_id'],
                #                      'lsh_title': batch['lsh_title'], 'org_title': batch['org_lsh_title']}
                #         # 之所以字典的key都用lsh开头，是因为模型topo中设置的输入就是以lsh开头的
                #         rsh_batch = {'lsh_frame_feature': batch['rsh_frame_feature'], 'tag_id': batch['rsh_tag_id'],
                #                      'id': batch['rsh_id'],
                #                      'lsh_title': batch['rsh_title'], 'org_title': batch['org_rsh_title']}
                #         sim_score = torch.Tensor(batch['similarity'])
                #         lsh_inputs = self.get_inputs_batch(lsh_batch)
                #         rsh_inputs = self.get_inputs_batch(rsh_batch)
                #         lsh_label_keys, lsh_labels = self.get_label_batch(lsh_batch)
                #         rsh_label_keys, rsh_labels = self.get_label_batch(rsh_batch)
                #         lsh_inputs[lsh_label_keys[0]] = lsh_labels[0]
                #         rsh_inputs[rsh_label_keys[0]] = rsh_labels[0]
                #
                #         (_, lsh_embedding), loss_1 = self.model(lsh_inputs)
                #         (_, rsh_embedding), loss_2 = self.model(rsh_inputs)
                #
                #         sim_score_predict = torch.sum(lsh_embedding * rsh_embedding, dim=1)
                #         sim_score = sim_score.to(device)
                #
                #         # loss_3 = loss_func(sim_score_predict, sim_score)
                #
                #         sim_score_predict = sim_score_predict.detach().cpu().numpy()
                #         sim_score = sim_score.detach().cpu().numpy()
                #         sim_score_predict_list.extend(sim_score_predict.astype(np.float16).tolist())
                #         sim_score_label_list.extend(sim_score.astype(np.float16).tolist())
                #
                #     # logging.info(
                #     #     'epoch: {}/{}   val batch_average loss: {}'.format(epoch, self.cfg.TRAINING.EPOCHS), loss/(step+1))
                #     spearman = scipy.stats.spearmanr(sim_score_predict_list, sim_score_label_list).correlation
                #     logging.info('epoch {}/{}    val spearman score: {}'.format(epoch, self.cfg.TRAINING.EPOCHS, spearman))

            # 一个epoch结束后验证
            self.model.eval()
            sim_score_predict_list = []
            sim_score_label_list = []
            for step, batch in tqdm.tqdm(enumerate(val_dataset_loader)):
                lsh_batch = {'lsh_frame_feature': batch['lsh_frame_feature'], 'tag_id': batch['lsh_tag_id'],
                             'id': batch['lsh_id'],
                             'lsh_title': batch['lsh_title'], 'org_title': batch['org_lsh_title']}
                # 之所以字典的key都用lsh开头，是因为模型topo中设置的输入就是以lsh开头的
                rsh_batch = {'lsh_frame_feature': batch['rsh_frame_feature'], 'tag_id': batch['rsh_tag_id'],
                             'id': batch['rsh_id'],
                             'lsh_title': batch['rsh_title'], 'org_title': batch['org_rsh_title']}
                sim_score = torch.Tensor(batch['similarity'])
                lsh_inputs = self.get_inputs_batch(lsh_batch)
                rsh_inputs = self.get_inputs_batch(rsh_batch)
                lsh_label_keys, lsh_labels = self.get_label_batch(lsh_batch)
                rsh_label_keys, rsh_labels = self.get_label_batch(rsh_batch)
                lsh_inputs[lsh_label_keys[0]] = lsh_labels[0]
                rsh_inputs[rsh_label_keys[0]] = rsh_labels[0]

                (_, lsh_embedding), loss_1 = self.model(lsh_inputs)
                (_, rsh_embedding), loss_2 = self.model(rsh_inputs)

                sim_score_predict = torch.sum(lsh_embedding * rsh_embedding, dim=1)
                sim_score = sim_score.to(device)

                # loss_3 = loss_func(sim_score_predict, sim_score)

                sim_score_predict = sim_score_predict.detach().cpu().numpy()
                sim_score = sim_score.detach().cpu().numpy()
                sim_score_predict_list.extend(sim_score_predict.astype(np.float16).tolist())
                sim_score_label_list.extend(sim_score.astype(np.float16).tolist())

            # logging.info(
            #     'epoch: {}/{}   val batch_average loss: {}'.format(epoch, self.cfg.TRAINING.EPOCHS), loss/(step+1))
            spearman = scipy.stats.spearmanr(sim_score_predict_list, sim_score_label_list).correlation
            logging.info('epoch {}/{}    val spearman score: {}'.format(epoch, self.cfg.TRAINING.EPOCHS, spearman))

            # if 'SPEARMAN_EVAL' in self.cfg.DATASET:  # run spearman test if eval if SPEARMAN_EVAL config is found
            #     self.evaluate_spearman(dataset_key='SPEARMAN_EVAL')
            # if self.eval_dataloader:  # run eval
            #     self.eval_model(epoch)
            # self.save_model(epoch)

            tmp_epoch_model_file = sys_tmpfile.get_temp_file_path_once()
            save_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint",
                                           "Epoch_{}_{:.4f}.bin".format(
                                               epoch, spearman))
            # save_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint",
            #                                "Epoch_{epoch}.bin".format(
            #                                    epoch))
            torch_nn_convertor.TorchNNConvertor.save_model(self.model, None, tmp_epoch_model_file)
            storage.put_storage_file(tmp_epoch_model_file, save_model_path)


    def evaluate_checkpoint(self, checkpoint_file: str, dataset_key: str, to_save_file):
        """
        :param checkpoint_file: the checkpoint used to evaluate the dataset
        :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
        :param to_save_file: the file to save the result
        :return:
        """
        assert dataset_key in self.cfg.DATASET
        dataset_config = self.cfg.DATASET[dataset_key]
        dataset_loader = self.gen_dataloader(dataset_config, training=False)
        self.load_checkpoint_for_eval(checkpoint_file)
        self.model.eval()
        id_list = []
        embedding_list = []
        for step, batch in tqdm.tqdm(enumerate(dataset_loader)):
            inputs = self.get_inputs_batch(batch)
            inputs = {'lsh_frame_feature': inputs['frame_feature'], 'lsh_title': inputs['title']}
            ids = batch['id']
            (logits, embedding), _ = self.model(inputs)
            embedding = embedding.detach().cpu().numpy()
            id_list.extend(list(ids))
            embedding_list.extend(embedding.astype(np.float16).tolist())
        output_res = dict(zip(id_list, embedding_list))
        with open(to_save_file, 'w') as f:
            json.dump(output_res, f)

    def evaluate_spearman(self, checkpoint_file='', dataset_key="SPEARMAN_EVAL"):
        """
        :param checkpoint_file: the checkpoint used to evaluate the dataset
        :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
        :return:
        """
        if checkpoint_file:
            self.load_checkpoint_for_eval(checkpoint_file)
        self.model.eval()
        dataset_config = self.cfg.DATASET[dataset_key]
        dataset_loader = self.gen_dataloader(dataset_config, training=False)
        id_list = []
        embedding_list = []
        for step, batch in tqdm.tqdm(enumerate(dataset_loader)):
            inputs = self.get_inputs_batch(batch)
            inputs = {'lsh_frame_feature': inputs['frame_feature'], 'lsh_title': inputs['title']}
            ids = batch['id']
            (logits, embedding), _ = self.model(inputs)
            embedding = embedding.detach().cpu().numpy()
            embedding_list.append(embedding)
            id_list += ids
        embeddings = np.concatenate(embedding_list)
        embedding_map = dict(zip(id_list, embeddings))
        annotate = {}
        label_file = storage.get_storage_file(dataset_config['LABEL_FILE'])
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rk1, rk2, score = line.split('\t')
                annotate[(rk1, rk2)] = float(score)
        sim_res = []
        logging.info('num embedding: {}, num annotates: {}'.format(len(embedding_map), len(annotate)))
        for (k1, k2), v in annotate.items():
            if k1 not in embedding_map or k2 not in embedding_map:
                continue
            sim_res.append((v, (embedding_map[k1] * embedding_map[k2]).sum()))
        spearman = scipy.stats.spearmanr([x[0] for x in sim_res], [x[1] for x in sim_res]).correlation
        logging.info('spearman score: {}'.format(spearman))
        self.temporary_map['spearman'] = spearman

    def save_model(self, epoch):
        self.model.eval()
        self.temporary_map.update(self.eval_data[-1])  # update temporary_map with latest eval info
        tmp_epoch_model_file = sys_tmpfile.get_temp_file_path_once()
        save_model_path = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint",
                                       "Epoch_{epoch}_{precision:.4f}_{recall:.4f}_{spearman:.4f}.bin".format_map(
                                           self.temporary_map))
        torch_nn_convertor.TorchNNConvertor.save_model(self.model, None, tmp_epoch_model_file)
        storage.put_storage_file(tmp_epoch_model_file, save_model_path)

    def train(self, checkpoint_file='', dataset_key="EVAL_DATA"):

        self.save_config_file()
        if checkpoint_file:
            logging.info('Loading pretrained model...')
            self.load_checkpoint_for_finetune(checkpoint_file)
            logging.info('Finished!')
        for epoch in range(1, self.cfg.TRAINING.EPOCHS + 1):
            logging.info("Training Epoch: " + str(epoch).center(60, "="))
            self.train_epoch()
            if 'SPEARMAN_EVAL' in self.cfg.DATASET:  # run spearman test if eval if SPEARMAN_EVAL config is found
                self.evaluate_spearman(dataset_key='SPEARMAN_EVAL')
            if self.eval_dataloader:  # run eval
                self.eval_model(epoch)
            self.save_model(epoch)

    def load_checkpoint_for_finetune(self, checkpoint_file):
        """
        :param checkpoint_file: checkpoint file used to eval model
        :return:
        """
        save_model_dir = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint")
        model_path = os.path.join(save_model_dir, checkpoint_file)
        model_file = storage.get_storage_file(model_path)
        model = torch.load(model_file)
        # print(model)
        # print(self.model)
        self.model.load_state_dict(model.state_dict())


        # 只加载representation部分模型

        # # print(model.representation_model_arr)
        # # print(self.model.module.representation_model_arr)
        # self.model.module.representation_model_arr.load_state_dict(model.representation_model_arr.state_dict())
        # self.init_gpu_setting()
        # self.model.train()

    def load_checkpoint_for_eval(self, checkpoint_file):
        """
        :param checkpoint_file: checkpoint file used to eval model
        :return:
        """
        save_model_dir = os.path.join(self.cfg.RUNTIME.SAVE_MODEL_DIR, "checkpoint")
        model_path = os.path.join(save_model_dir, checkpoint_file)
        model_file = storage.get_storage_file(model_path)
        self.model = torch.load(model_file)
        self.init_gpu_setting()
        self.model.eval()

    def empty_loop_test(self):
        """
        :return: empty loop to test IO speed
        """
        for _ in tqdm.tqdm(self.train_dataloader):
            continue
