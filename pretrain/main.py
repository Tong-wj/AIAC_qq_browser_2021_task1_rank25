import argparse
import logging
import os

from lichee import plugin

from module.models import ConcatCls, EmbeddingTrainer
from module.feature_parser import FrameFeature, TagParser, VidParser
from module.utils import LayeredOptim, BCELoss, PRScore


def parse_args():
    parser = argparse.ArgumentParser(description="QQ Browser video embedding challenge")

    # parser.add_argument('--mode', type=str, default='data_prepare', help='choose mode from [train, test, eval]')
    # parser.add_argument("--dataset", type=str, default='DATA_PREPARE')
    # parser.add_argument("--model_config_file", type=str, default="embedding_split_data_prepare.yaml",
    #                     help="The path of configuration json file.")

    parser.add_argument('--mode', type=str, default='train', help='choose mode from [train, test, eval]')
    parser.add_argument("--dataset", type=str, default='EVAL_DATA')
    parser.add_argument("--model_config_file", type=str, default="embedding_train.yaml",
                        help="The path of configuration json file.")

    # parser.add_argument('--mode', type=str, default='finetune', help='choose mode from [train, test, eval]')
    # parser.add_argument("--dataset", type=str, default='FINETUNE')
    # parser.add_argument("--model_config_file", type=str, default="embedding_split_finetune_half.yaml",
    #                     help="The path of configuration json file.")
    # # parser.add_argument("--model_config_file", type=str, default="embedding_split_finetune_full.yaml",
    # #                     help="The path of configuration json file.")

    # parser.add_argument('--mode', type=str, default='test', help='choose mode from [train, test, eval]')
    # # parser.add_argument("--dataset", type=str, default='SPEARMAN_TEST_A')
    # parser.add_argument("--dataset", type=str, default='SPEARMAN_TEST_B')
    # parser.add_argument("--model_config_file", type=str, default="embedding_train.yaml",
    #                     help="The path of configuration json file.")



    # parser.add_argument("--model_config_file", type=str, default="embedding_split_data_prepare.yaml",
    #                     help="The path of configuration json file.")
    # parser.add_argument("--model_config_file", type=str, default="embedding_split_finetune_full.yaml",
    #                     help="The path of configuration json file.")
    # parser.add_argument("--model_config_file", type=str, default="embedding_split_finetune_half.yaml",
    #                     help="The path of configuration json file.")
    # parser.add_argument("--model_config_file", type=str, default="embedding_train.yaml",
    #                     help="The path of configuration json file.")
    # parser.add_argument("--model_config_file", type=str, default="embedding_eval.yaml",
    #                     help="The path of configuration json file.")


    # pretrained models
    parser.add_argument('--checkpoint', type=str, default='Epoch_3_0.7847_0.4737_0.6564.bin')
    # parser.add_argument('--checkpoint', type=str, default='Epoch_3_0.7872_0.4716_0.6535.bin')
    # parser.add_argument('--checkpoint', type=str, default='Epoch_3_0.7596_0.4386_0.6600.bin')
    # parser.add_argument('--checkpoint', type=str, default='Epoch_15_0.6622_0.4975_0.6432.bin')
    # parser.add_argument('--checkpoint', type=str, default='Epoch_30_0.8657_0.8177_0.6321.bin')

    # finetuned models

    # parser.add_argument('--checkpoint', type=str, default='Epoch_16_0.9461.bin')

    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_1_0.7149_0.3580_0.8861.bin')  # val 0.8861    test 0.791761
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_2_0.6834_0.3865_0.9182.bin')  # val 0.9182    test 0.794771
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_3_0.6681_0.3990_0.9321.bin')  # val 0.9321    test 0.797128

    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_4_0.6520_0.4104_0.9414.bin')  # val 0.9414    test 0.793452
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_5_0.6460_0.4170_0.9458.bin')  # val 0.9458    test 0.795272
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_6_0.6343_0.4214_0.9458.bin')  # val 0.9458    test 0.797054
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_7_0.6273_0.4250_0.9478.bin')  # val 0.9478    test 0.796594


    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_3_0.6517_0.4377_0.9239.bin')  # val 0.9239    test 0.793332
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_6_0.6366_0.4440_0.9406.bin')  # val 0.9406    test 0.788933
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_10_0.6126_0.4571_0.9440.bin')  # val 0.9440    test 0.


    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_3_0.6545_0.3934_0.9306.bin')  # val 0.9306    test 0.796300

    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_4_0.6748_0.5959_0.9372.bin')  # val 0.9372    test 0.

    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_1_0.7084_0.3524_0.8561.bin')  # val 0.8199    test 0.783425
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_3_0.6788_0.3800_0.8870.bin')  # val 0.8350    test  0.788383
    # parser.add_argument('--checkpoint', type=str,
    #                     default='Epoch_8_0.6174_0.4141_0.8905.bin')  # val 0.8374  test 0.785597
    # parser.add_argument('--checkpoint', type=str, default='Epoch_16_0.0005_0.5401_0.8150.bin')    # test  0.727105
    # parser.add_argument('--checkpoint', type=str, default='Epoch_17_0.0005_0.5233_0.8807.bin')    # tsst  0.734024
    # parser.add_argument('--checkpoint', type=str, default='Epoch_18_0.0005_0.5314_0.9159.bin')    # test  0.737027
    # parser.add_argument('--checkpoint', type=str, default='Epoch_19_0.0005_0.5270_0.9310.bin')    # test  0.740912
    # parser.add_argument('--checkpoint', type=str, default='Epoch_26_0.0005_0.5347_0.9421.bin')    # test  0.737266
    # parser.add_argument('--checkpoint', type=str, default='Epoch_30_0.0005_0.5274_0.9444.bin')    # test  0.736616

    parser.add_argument("--trainer", type=str, default="embedding_trainer",
                        help="choose trainer, you can run --show_trainer list all supported trainer")
    parser.add_argument("--show", type=str,
                        help="list all supported module, [trainer|target|model|loss|optim|lr_schedule]")
    parser.add_argument('--to-save-file', type=str, default='result.json')

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    args = parse_args()
    trainer = plugin.get_plugin(plugin.PluginType.TRAINER, args.trainer)
    if args.mode == 'train':
        trainer = trainer(args.model_config_file)
        trainer.train()
    elif args.mode == 'data_prepare':
        trainer = trainer(args.model_config_file, init_model=False)
        trainer.pairwise_data_prepare(dataset_key=args.dataset, checkpoint_file=args.checkpoint)
    elif args.mode == 'finetune':
        trainer = trainer(args.model_config_file)
        trainer.finetune_with_spearman(dataset_key=args.dataset, checkpoint_file=args.checkpoint)
    elif args.mode == 'test':
        trainer = trainer(args.model_config_file, init_model=False)
        if os.path.exists(args.to_save_file):
            raise FileExistsError('to save file: {} already existed'.format(args.to_save_file))
        trainer.evaluate_checkpoint(
            dataset_key=args.dataset, checkpoint_file=args.checkpoint, to_save_file=args.to_save_file)
    else:
        trainer = trainer(args.model_config_file, init_model=False)
        trainer.evaluate_spearman(dataset_key=args.dataset, checkpoint_file=args.checkpoint)
