import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer

def main():
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = None 
    val_data_loader = DataFactory.get_data_loader(config, split_type=config.qbnorm_mode)
    test_data_loader = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, None,
                      config=config,
                      train_data_loader=val_data_loader,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer,
                      qb_norm='QB-Norm')

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")
    trainer.validate_qbnorm()


if __name__ == '__main__':
    main()

