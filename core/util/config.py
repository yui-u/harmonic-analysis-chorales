from datetime import datetime
from pathlib import Path


class ConfigBase:
    @property
    def data_suffix(self):
        raise NotImplementedError

    @property
    def model_filename_format(self):
        raise NotImplementedError

    def get_model_filename(self, epoch):
        raise NotImplementedError

    @property
    def log_filename(self):
        raise NotImplementedError

    @property
    def model_suffix(self):
        raise NotImplementedError

    def write_log(self, logger):
        raise NotImplementedError


class Config(ConfigBase):
    def __init__(self, args):
        self.now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        for k, v in vars(args).items():
            setattr(self, k, v)
        if '4part' in self.dataset:
            self.cv_num_set = -1
            self.cv_set_no = -1
        self.dir_output = self.dir_instance_output / Path('{}-{}'.format(self.model_suffix, self.data_suffix))

    @property
    def data_suffix(self):
        suf = '{}-{}'.format(self.dataset, self.key_preprocessing)
        if self.dataset in ['bach60']:
            suf += '-cv{}-set{}'.format(self.cv_num_set, self.cv_set_no)
        return suf

    def get_model_filename(self, epoch):
        return '{}-{}-seed{}-ep{}-{}.model'.format(
            self.model_suffix.lower(), self.data_suffix.lower(), self.seed, epoch, self.now)

    @property
    def log_filename(self):
        return '{}-{}-seed{}-{}.log'.format(
            self.model_suffix.lower(), self.data_suffix.lower(), self.seed, self.now)

    @property
    def model_suffix(self):
        suf = '{}-{}-m{}-q{}-shift{}'.format(
            self.activation_fn, self.metric, self.num_modes, self.quality_magnification,
            str(not self.no_shift).lower()
        )
        return suf

    def write_log(self, logger):
        raise NotImplementedError
