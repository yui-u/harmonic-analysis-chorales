from pathlib import Path

import torch
from core.common.constants import *
from core.preprocess.instances import ListInstance, ValueInstance


PAD_CHROMA = [0] * 12
PAD_LABEL = '<PAD>'


class Bach60Reader(object):
    def __init__(self, config):
        self._config = config
        self._key_preprocessing = config.key_preprocessing
        self._folds = self._read_folds()

    def _read_folds(self):
        folds = []
        assert self._config.cv_num_set == 10
        fold_dir_path = Path('bach60/folds')
        for i in range(self._config.cv_num_set):
            test_fold_file = fold_dir_path / Path('test{}.txt'.format(i + 1))
            folds.append([])
            with test_fold_file.open('r') as f:
                for line in f:
                    cid = line.rstrip().split('/')[-1]
                    assert cid.endswith('.xml')
                    folds[i].append(cid.split('annotated')[0])
        return folds

    def _normalize_chroma_imperfect(self, chroma_list):
        # Since the key signature of the original score is not known,
        # we normalize it to have the fewest number of accidentals.
        chroma_sum = torch.tensor(chroma_list).sum(dim=0)
        expanded_chroma_sum = []
        for i in range(12):
            expanded_chroma_sum.append(torch.roll(chroma_sum, shifts=i, dims=-1))
        expanded_chroma_sum = torch.stack(expanded_chroma_sum, dim=0)
        accidentals = expanded_chroma_sum[:, 1] + expanded_chroma_sum[:, 3] + expanded_chroma_sum[:, 6] + expanded_chroma_sum[:, 8] + expanded_chroma_sum[:, 10]
        shift = accidentals.argmin(dim=0).item()
        shifted_chroma = torch.roll(torch.tensor(chroma_list), shifts=shift, dims=-1)
        return shift, shifted_chroma.tolist()

    def _add_instance(self, choral_id, raw_list, instances, shift=None, shifted_chroma=None):
        if shifted_chroma is not None:
            _chroma = shifted_chroma
        else:
            _chroma = raw_list['chroma'].copy()

        sequence_length = len(raw_list['chroma'])
        if sequence_length <= self._config.max_sequence_length:
            _chroma += [
                PAD_CHROMA for _ in range(self._config.max_sequence_length - sequence_length)]
        else:
            _chroma = raw_list['chroma'][:self._config.max_sequence_length]
        instances.append({
            'observation_chroma': ListInstance(list_instances=_chroma),
            'sequence_length': ValueInstance(sequence_length),
            META_DATA: {
                'reader_name': self.__class__.__name__,
                'choral_id': choral_id,
                'event_number': raw_list['event_number'],
                'bass': raw_list['bass'],
                'chord_label': raw_list['chord_label'],
                'melisma_meter': raw_list['melisma_meter'],
                'raw_chroma': raw_list['chroma'],
                'shift': shift
            }
        })

    def create_instance(self, logger):
        data_path = Path('bach60/jsbach_chorals_harmony.data')
        instances = []
        raw_list = {
            'event_number': [],
            'chroma': [],
            'bass': [],
            'melisma_meter': [],
            'chord_label': []
        }
        prev_choral_id = None
        with data_path.open('r') as f:
            for line in f:
                line = line.strip().split(',')
                choral_id = line[0]
                if prev_choral_id is None:
                    prev_choral_id = choral_id
                if prev_choral_id != choral_id:
                    logger.info('processed {}'.format(choral_id))
                    if self._key_preprocessing == KEY_PREPROCESS_NORMALIZE:
                        shift, shifted_chroma = self._normalize_chroma_imperfect(raw_list['chroma'])
                        self._add_instance(prev_choral_id, raw_list, instances, shift=shift, shifted_chroma=shifted_chroma)
                    elif self._key_preprocessing == KEY_PREPROCESS_AUGMENTED:
                        raw_chroma_tensor = torch.tensor(raw_list['chroma'])
                        for shift in range(12):
                            shifted_chroma = torch.roll(raw_chroma_tensor, shifts=shift, dims=-1).tolist()
                            self._add_instance(prev_choral_id, raw_list, instances, shift=shift, shifted_chroma=shifted_chroma)
                    else:
                        assert self._key_preprocessing == KEY_PREPROCESS_NONE
                        self._add_instance(prev_choral_id, raw_list, instances)
                    raw_list = {
                        'event_number': [],
                        'chroma': [],
                        'bass': [],
                        'melisma_meter': [],
                        'chord_label': []
                    }

                raw_list['event_number'].append(line[1])
                raw_list['chroma'].append([1 if c == 'YES' else 0 for c in line[2:14]])
                raw_list['bass'].append(line[14])
                raw_list['melisma_meter'].append(int(line[15]))
                raw_list['chord_label'].append(line[16].lstrip())

                prev_choral_id = choral_id
        if bool(raw_list['event_number']):
            self._add_instance(prev_choral_id, raw_list, instances)
        logger.info('read {} chorales'.format(len(instances)))
        return instances

    def split_train_dev_test(self, instances, logger):
        assert self._config.cv_num_set == 10
        train_instances = []
        dev_instances = []
        test_instances = []
        test_pieces = self._folds[self._config.cv_set_no]
        dev_pieces = self._folds[(self._config.cv_set_no + 1) % self._config.cv_num_set]
        dev_pieces += self._folds[(self._config.cv_set_no + 2) % self._config.cv_num_set]
        for instance in instances:
            choral_id = instance[META_DATA]['choral_id']
            if choral_id in test_pieces:
                test_instances.append(instance)
            elif choral_id in dev_pieces:
                dev_instances.append(instance)
            else:
                train_instances.append(instance)
        logger.info("{}-pieces={}, {}-pieces={}, {}-pieces={}".format(
            TRAIN, len(train_instances), DEV, len(dev_instances), TEST, len(test_instances)))
        return train_instances, dev_instances, test_instances

