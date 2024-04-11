import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)

        # data parameters
        parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='data/MSRVTT/vids', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=30)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=50, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=20, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=2, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=False)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=256, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--qbnorm_k', type=int, default=1)
        parser.add_argument('--qbnorm_beta', type=int, default=50)
        parser.add_argument('--qbnorm_mode', type=str, default='train')
        parser.add_argument('--mode', type=str, default='none')
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--threshold', type=float, default=0.1)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=50)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=4, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in multi-headed attention')
        parser.add_argument('--dropout1', type=float, default=0.3, help='Dropout prob. in the transformer pooling')
        parser.add_argument('--dropout2', type=float, default=0.6, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
