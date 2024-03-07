import math
import torch
from torch.utils.data import DataLoader, RandomSampler

from core.common.constants import *
from core.preprocess.dataset import CustomCollate, CustomDataset
from core.preprocess.instances import batch_to_device


class Evaluator(object):
    @staticmethod
    def evaluate(
            instances,
            model,
            config,
            logger,
            viterbi_output=False,
            metadata_output=False
    ):
        eval_data = CustomDataset(data=instances)

        sampler = RandomSampler(eval_data)

        batch_size = config.batch_size
        data_loader = DataLoader(dataset=eval_data, sampler=sampler, batch_size=batch_size,
                                 collate_fn=CustomCollate.collate, pin_memory=True)

        logger.info('***** Evaluating *****')
        model.zero_grad()
        total_loss = 0.0
        total_nll = 0.0
        total_items = 0
        total_tokens = 0
        total_output = []

        for _, batch in enumerate(data_loader):
            batch = batch_to_device(batch, config.device)
            model.eval()
            with torch.no_grad():
                output = model(batch, viterbi=viterbi_output)

                total_loss += output[LOCAL_LOSS].detach().item()
                total_nll += (-output[LOG_LIKELIHOOD]).detach().sum().item()

                total_items += output[BATCH_SIZE]
                total_tokens += batch['sequence_length'].sum().item()

                if metadata_output:
                    for k, v in batch.items():
                        assert k not in output
                        output[k] = v
                total_output.append(output)

        perplexity = math.exp(total_nll / total_tokens)
        total_nll = total_nll / total_items
        total_loss = total_loss / total_items

        eval_label = 'Eval-Loss'
        logger.info('{}:{}, nll:{}, perplexity:{}'.format(
            eval_label, total_loss, total_nll, perplexity))

        eval_output = {
            TOTAL_LOSS: total_loss,
            NLL: total_nll,
            PERPLEXITY: perplexity,
            OUTPUT: total_output,
        }
        return eval_output


