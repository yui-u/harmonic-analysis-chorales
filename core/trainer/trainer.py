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
    def save_model(config, model, epoch, accepted_num_modes):
        model_filename = config.get_model_filename(epoch)
        output_file = config.dir_output / Path(model_filename)
        if accepted_num_modes is not None:
            assert model.num_modes == accepted_num_modes, (model.num_modes, accepted_num_modes)
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
        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        torch.autograd.set_detect_anomaly(True)
        model.train()
        best_eval_metric = None
        best_epoch = -1
        best_model = None
        warmup_count = 0
        for epoch in iterator:
            logger.info('***** Epoch: {} *****'.format(epoch))
            warmup_count += 1
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
            eval_m_result = Evaluator.evaluate(dev_instances, model, config, logger)
            eval_m_metric = eval_m_result[config.metric]
            if config.dynamic_num_modes:
                if model.num_modes <= 1:
                    eval_metric = eval_m_metric
                    if config.warmup_num_modes <= warmup_count:
                        model.num_modes = model.num_modes + 1
                        logger.info('Automatically increase num_modes from 1 to 2 to apply dynamic num_modes.')
                        warmup_count = 0  # reset warmup count
                else:
                    eval_mm1_result = Evaluator.evaluate(
                        dev_instances, model, config, logger, num_modes_eval=model.num_modes - 1)
                    eval_mm1_metric = eval_mm1_result[config.metric]
                    if config.warmup_num_modes <= warmup_count:
                        stationary_distributions = model.get_hsmm_params()['transition_stationary'][:, :-1]  # (M, C+1) => (M, C), remove rest
                        max_cossim = max(
                            [cos_sim(stationary_distributions[pmi], stationary_distributions[-1]) for pmi in
                             range(model.num_modes - 1)])

                        if lower_is_better:
                            metric_ratio = (eval_mm1_metric - eval_m_metric) / eval_m_metric
                        else:
                            metric_ratio = (eval_m_metric - eval_mm1_metric) / eval_m_metric

                        if (config.acceptance_th <= metric_ratio) and (max_cossim <= config.cossim_limit):
                            model.num_modes = model.num_modes + 1
                            eval_metric = eval_m_metric
                            logger.info('Loss-ratio:{}, Max-cossim: {}, Increment the number of modes'.format(
                                metric_ratio, max_cossim))
                            warmup_count = 0  # reset warmup count
                        else:
                            eval_metric = eval_mm1_metric
                            logger.info('Loss-ratio:{}, Max-cossim: {}'.format(metric_ratio, max_cossim))
                    else:
                        eval_metric = eval_mm1_metric
            else:
                eval_metric = eval_m_metric

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
                if config.dynamic_num_modes:
                    best_model.num_modes = max(best_model.num_modes - 1, 1)  # Save confirmed num_modes (last mode added not yet accepted)
            else:
                if (0 < config.patience) and (config.patience < epoch - best_epoch):
                    logger.info('Early stopping, Best Epoch: {}'.format(best_epoch))
                    break

        if config.dynamic_num_modes:
            logger.info('End Training, Best Epoch: {}, # Modes: {}'.format(best_epoch, best_model.num_modes))
        else:
            logger.info('End Training, Best Epoch: {}'.format(best_epoch))
        model_filename = Trainer.save_model(
            config, best_model, best_epoch, accepted_num_modes=best_model.num_modes)
        return best_model, model_filename
