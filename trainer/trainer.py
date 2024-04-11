from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, beat_similarity, qb_norm
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None, qb_norm=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.qb_norm = qb_norm
        self.qbnorm_beta = config.qbnorm_beta
        self.qbnorm_k = config.qbnorm_k
        self.metric = config.metric


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):

            data['music'] = data['music'].to(self.device)
            data['video'] = data['video'].to(self.device)
            data['music_beat'] = data['music_beat'].to(self.device)
            data['video_beat'] = data['video_beat'].to(self.device)

            text_embeds, video_embeds = self.model(data, 'train')
            output = sim_matrix_training(text_embeds['music_fuse'], video_embeds['video_fuse'])
            output1 = sim_matrix_training(text_embeds['music_beat'], video_embeds['video_beat'])

            loss = self.loss(output, self.model.clip_logit_scale) + self.beta * self.loss(output1, self.model.clip_logit_scale)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip_logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['AVG-window'] > self.best_window:
                    self.best_window = val_res['AVG-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['AVG'] > self.best:
                    self.best = val_res['AVG']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        music_embed_arr, vid_embed_arr = [], []
        mus_beat_arr, vid_beat_arr = [], []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):

                data['video'] = data['video'].to(self.device)
                data['music'] = data['music'].to(self.device)
                data['music_beat'] = data['music_beat'].to(self.device)
                data['video_beat'] = data['video_beat'].to(self.device)

                music_embed, vid_embed = self.model(data)

                mus_beat_arr.append(music_embed['music_beat'])
                vid_beat_arr.append(vid_embed['video_beat'])
                music_embed_arr.append(music_embed['music_fuse'])
                vid_embed_arr.append(vid_embed['video_fuse'])
                sims_batch = sim_matrix_training(music_embed['music_fuse'], vid_embed['video_fuse'], self.pooling_type)
                sims_batch1 = sim_matrix_training(music_embed['music_beat'], vid_embed['video_beat'], self.pooling_type) 

                curr_loss = self.loss(sims_batch, self.model.clip_logit_scale) + self.loss(sims_batch1, self.model.clip_logit_scale) * self.beta
                total_val_loss += curr_loss.item()
                
            music_embeds = torch.cat(music_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            mus_beats = torch.cat(mus_beat_arr)
            vid_beats = torch.cat(vid_beat_arr)
            sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds)
                    
            sims = sims.cpu().detach()
            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"R@50: {res['R50']} (window: {res['R50-window']})\n",
                  f"R@100: {res['R100']} (window: {res['R100-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"Means: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"AVG: {res['AVG']} (window: {res['AVG-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res

    def _valid_qbnorm_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        music_embed_arr, vid_embed_arr = [], []
        mus_beat_arr, vid_beat_arr = [], []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):

                data['video'] = data['video'].to(self.device)
                data['music'] = data['music'].to(self.device)
                data['music_beat'] = data['music_beat'].to(self.device)
                data['video_beat'] = data['video_beat'].to(self.device)
                music_embed, vid_embed = self.model(data)

                mus_beat_arr.append(music_embed['music_beat'])
                vid_beat_arr.append(vid_embed['video_beat'])
                music_embed_arr.append(music_embed['music_fuse'])
                vid_embed_arr.append(vid_embed['video_fuse'])
                sims_batch = sim_matrix_training(music_embed['music_fuse'], vid_embed['video_fuse'], self.pooling_type) + \
                                sim_matrix_training(music_embed['music_beat'], vid_embed['video_beat'], self.pooling_type)

                curr_loss = self.loss(sims_batch, self.model.clip_logit_scale)
                total_val_loss += curr_loss.item()

            music_embeds = torch.cat(music_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)
            mus_beats = torch.cat(mus_beat_arr)
            vid_beats = torch.cat(vid_beat_arr)

            if self.qb_norm is not None:
                if self.metric == 'v2t':
                    test_embed_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds).cpu().detach()
                    test_beat_sims = sim_matrix_inference(mus_beats.unsqueeze(1), vid_beats).cpu().detach()
                    vid_embed_arr = []
                    vid_beat_arr = []

                    for _, data in tqdm(enumerate(self.train_data_loader)):
                        data['video'] = data['video'].to(self.device)
                        data['music'] = data['music'].to(self.device)
                        data['music_beat'] = data['music_beat'].to(self.device)
                        data['video_beat'] = data['video_beat'].to(self.device)
                        _, vid_embed = self.model(data)
                        vid_embed_arr.append(vid_embed['video_fuse'])
                        vid_beat_arr.append(vid_embed['video_beat'])
                    vid_embeds = torch.cat(vid_embed_arr)
                    vid_beats = torch.cat(vid_beat_arr)

                    train_embed_sims = train_sims = sim_matrix_inference(vid_embeds.unsqueeze(1), music_embeds)
                    train_embed_sims = train_embed_sims.cpu().detach()
                    train_embed_sims, test_embed_sims = np.squeeze(np.array(train_embed_sims), axis=1), np.squeeze(np.array(test_embed_sims), axis=1)
                    test_embed_sims = np.transpose(test_embed_sims, (1, 0))
                    test_embed_sims = qb_norm(train_embed_sims, test_embed_sims, self.qbnorm_k, self.qbnorm_beta)
                    test_embed_sims = np.transpose(test_embed_sims, (1, 0))
                    test_embed_sims  = np.expand_dims(test_embed_sims , axis=1)
                    test_embed_sims = torch.from_numpy(test_embed_sims)

                    train_beat_sims = sim_matrix_inference(vid_beats.unsqueeze(1), mus_beats)
                    train_beat_sims = train_beat_sims.cpu().detach()
                    train_beat_sims, test_beat_sims = np.squeeze(np.array(train_beat_sims), axis=1), np.squeeze(np.array(test_beat_sims), axis=1)
                    test_beat_sims = np.transpose(test_beat_sims, (1, 0))
                    test_beat_sims = qb_norm(train_beat_sims, test_beat_sims, self.qbnorm_k, self.qbnorm_beta)
                    test_beat_sims = np.transpose(test_beat_sims, (1, 0))
                    test_beat_sims = np.expand_dims(test_beat_sims, axis=1)
                    test_beat_sims = torch.from_numpy(test_beat_sims)
                    
                    print(test_beat_sims.shape, test_beat_sims.shape)
                    # test_sims = torch.add(test_embed_sims, test_beat_sims * 0.8)
                    test_sims = test_embed_sims
                    print(test_sims.shape)
                   
                else:
                    test_embed_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds).cpu().detach()
                    test_beat_sims = sim_matrix_inference(mus_beats.unsqueeze(1), vid_beats).cpu().detach()
                    music_embed_arr = []
                    mus_beat_arr = []
                    # vid_embed_arr = []
                    for _, data in tqdm(enumerate(self.train_data_loader)):
                        data['video'] = data['video'].to(self.device)
                        data['music'] = data['music'].to(self.device)
                        data['music_beat'] = data['music_beat'].to(self.device)
                        data['video_beat'] = data['video_beat'].to(self.device)
                        music_embed, _ = self.model(data)
                        music_embed_arr.append(music_embed['music_fuse'])
                        mus_beat_arr.append(music_embed['music_beat'])
                        # vid_embed_arr.append(vid_embed)
                    music_embeds = torch.cat(music_embed_arr)
                    mus_beats = torch.cat(mus_beat_arr)
                    # vid_embeds = torch.cat(vid_embed_arr)
                    train_embed_sims = sim_matrix_inference(music_embeds.unsqueeze(1), vid_embeds)
                    train_embed_sims = train_embed_sims.cpu().detach()
                    train_embed_sims, test_embed_sims = np.squeeze(np.array(train_embed_sims), axis=1), np.squeeze(np.array(test_embed_sims), axis=1)
                    # train_sims, test_sims = np.transpose(train_sims, (1, 0)), np.transpose(test_sims, (1, 0))
                    test_embed_sims = qb_norm(train_embed_sims, test_embed_sims, self.qbnorm_k, self.qbnorm_beta)
                    # test_sims = np.transpose(test_sims, (1, 0))
                    test_embed_sims  = np.expand_dims(test_embed_sims , axis=1)
                    test_embed_sims = torch.from_numpy(test_embed_sims)
                    
                    train_beat_sims = sim_matrix_inference(mus_beats.unsqueeze(1), vid_beats)
                    train_beat_sims = train_beat_sims.cpu().detach()
                    train_beat_sims, test_beat_sims = np.squeeze(np.array(train_beat_sims), axis=1), np.squeeze(np.array(test_beat_sims), axis=1)
                    # train_sims, test_sims = np.transpose(train_sims, (1, 0)), np.transpose(test_sims, (1, 0))
                    test_beat_sims = qb_norm(train_beat_sims, test_beat_sims, self.qbnorm_k, self.qbnorm_beta)
                    # test_sims = np.transpose(test_sims, (1, 0))
                    test_beat_sims = np.expand_dims(test_beat_sims, axis=1)
                    test_beat_sims = torch.from_numpy(test_beat_sims)

                    test_sims = test_beat_sims + test_embed_sims  
                    print(test_beat_sims.shape, test_embed_sims.shape)
                    print(test_sims.shape)    
                    test_sims = test_embed_sims
                    print(test_sims.shape)                       
                
           
            sims = test_sims
            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"R@50: {res['R50']} (window: {res['R10-window']})\n",
                  f"R@100: {res['R100']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res
