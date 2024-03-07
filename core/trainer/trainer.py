import math
from pathlib import Path

import copy
import torch
from torch.nn.utils import clip_grad_value_
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange

from core.common.constants import *
from core.preprocess.dataset import CustomCollate, CustomDataset
from core.eval.evaluator import Evaluator
from core.preprocess.instances import batch_to_device


class Trainer(object):
    @staticmethod
    def save_model(config, model, epoch, dev_instances, logger):
        model_filename = config.get_model_filename(epoch)
        output_file = config.dir_output / Path(model_filename)

        checkpoint = {
            'model': model.state_dict()
        }

        torch.save(checkpoint, output_file)
        return model_filename

    @staticmethod
    def train(train_instances, dev_instances, model, config, logger):
        if config.metric in [NLL, TOTAL_LOSS, PERPLEXITY]:
            lower_is_better = True
        else:
            raise NotImplementedError

        train_data = CustomDataset(data=train_instances)

        sampler = RandomSampler(train_data)

        batch_size = config.batch_size
        iterator = trange(config.num_epochs, desc='Epoch', disable=False)
        data_loader = DataLoader(dataset=train_data, sampler=sampler, batch_size=batch_size,
                                 collate_fn=CustomCollate.collate, pin_memory=True)

        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        logger.info('***** Start Training *****')
        torch.autograd.set_detect_anomaly(True)
        model.train()
        best_eval_metric = None
        best_epoch = -1
        best_model = None
        for epoch in iterator:
            logger.info('***** Epoch: {} *****'.format(epoch))
            total_loss = 0.0
            total_nll = 0.0
            total_items = 0
            total_tokens = 0
            for _, batch in enumerate(data_loader):
                batch = batch_to_device(batch, config.device)
                model.to(config.device)

                model.train()
                model.zero_grad()

                output = model(batch)
                loss = output[LOCAL_LOSS] / output[BATCH_SIZE]
                loss.backward()
                total_loss += output[LOCAL_LOSS].detach().item()
                total_nll += (-output[LOG_LIKELIHOOD]).detach().sum().item()
                if 0 < config.gradient_clip_value:
                    clip_grad_value_(model.parameters(), config.gradient_clip_value)
                optimizer.step()

                total_tokens += batch['sequence_length'].sum().item()
                total_items += output[BATCH_SIZE]

            total_loss = total_loss / total_items
            perplexity = math.exp(total_nll / total_tokens)
            total_nll = total_nll / total_items
            logger.info(
                'Train-Loss:{}, nll:{}, perplexity:{}'.format(
                    total_loss, total_nll, perplexity))

            # eval
            eval_result = Evaluator.evaluate(dev_instances, model, config, logger)
            eval_metric = eval_result[config.metric]

            if best_eval_metric is None:
                model_to_be_updated = True
            elif lower_is_better and (eval_metric < best_eval_metric):
                model_to_be_updated = True
            elif (not lower_is_better) and (best_eval_metric < eval_metric):
                model_to_be_updated = True
            else:
                model_to_be_updated = False

            if model_to_be_updated:
                logger.info('Update best epoch')
                best_eval_metric = eval_metric
                best_epoch = epoch
                best_model = copy.deepcopy(model)
            else:
                if (0 < config.patience) and (config.patience < epoch - best_epoch):
                    logger.info('Early stopping, Best Epoch: {}'.format(best_epoch))
                    break

        logger.info('End Training, Best Epoch: {}'.format(best_epoch))
        model_filename = Trainer.save_model(config, best_model, best_epoch, dev_instances, logger)
        return best_model, model_filename
