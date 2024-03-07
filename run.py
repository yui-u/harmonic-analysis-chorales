import argparse
import math
import pickle
from pathlib import Path

import pandas as pd
import torch

from core.common.constants import *
from core.eval.evaluator import Evaluator
from core.model.model import NeuralHSMM, INVERSION_DICT, PITCH_TO_DEGREE
from core.postprocess.visualize_hmm import HmmVisualizer
from core.preprocess.reader_bach60 import Bach60Reader
from core.preprocess.reader_fourpart_chorale import BachChoraleReader
from core.trainer.trainer import Trainer
from core.util.config import Config
from core.util.logging import create_logger
from core.util.util import set_seed


def add_arguments():
    parser = argparse.ArgumentParser(prog='run_nhsmm_tonicmodal')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='`cuda:n` where `n` is an integer, or `cpu`')
    parser.add_argument('--dataset',
                        type=str,
                        default='bach60',
                        choices=['bach60', 'bach-4part-mxl'],
                        help='Dataset name')
    parser.add_argument('--dir_instance_output',
                        type=str,
                        default='out',
                        help='output directory path')
    parser.add_argument('--cv_num_set',
                        type=int,
                        default=10,
                        help='Number of cross-validation sets')
    parser.add_argument('--cv_set_no',
                        type=int,
                        default=0,
                        help='Cross-validation set number')
    parser.add_argument('--key_preprocessing',
                        type=str,
                        choices=[KEY_PREPROCESS_NONE, KEY_PREPROCESS_NORMALIZE],
                        default=KEY_PREPROCESS_NORMALIZE,
                        help='key pre-processing type')
    parser.add_argument('--pivot_chord_selection',
                        type=str,
                        choices=['first', 'last'],
                        default='last',
                        help='In the case of a pivot code, the selected record is evaluated.')
    parser.add_argument('--max_sequence_length',
                        type=int,
                        default=512,
                        help='maximum sequence length')
    parser.add_argument('--model_to_initialize',
                        type=str,
                        default=None,
                        help='Pretrained model filename')
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help='seed')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='batch size')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=480,
                        help='maximum epochs')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='learning rate')
    parser.add_argument('--max_residential_time',
                        type=int,
                        default=16,
                        help='max residential time')
    parser.add_argument('--num_modes',
                        type=int,
                        default=2,
                        help='Number of modes')
    parser.add_argument('--no_shift',
                        action='store_true',
                        help='Disable root shift')
    parser.add_argument('--inter_key_transition_limit',
                        type=float,
                        default=0.01,
                        help='inter-key transition probability limit.')
    parser.add_argument('--metric',
                        type=str,
                        default=NLL,
                        help='Metric to update the best model')
    parser.add_argument('--activation_fn',
                        type=str,
                        choices=['tanh', 'relu', 'lrelu', 'mish'],
                        default='tanh',
                        help='Type of activation function')
    parser.add_argument('--num_encoder_rnn_layers',
                        type=int,
                        default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--mlp_hidden_size',
                        type=int,
                        default=16,
                        help='Number of hidden states.')
    parser.add_argument('--quality_magnification',
                        type=float,
                        default=5.0,
                        help='Magnification to control occurrence/non-occurrence in the quality template (before applying sigmoid)')
    parser.add_argument('--gradient_clip_value',
                        type=float,
                        default=-1,
                        help='gradient clip value. Set -1 to disable')
    parser.add_argument('--dropout_p',
                        type=float,
                        default=0.125,
                        help='dropout proportion')
    parser.add_argument('--patience',
                        type=int,
                        default=80,
                        help='early stop patience. set -1 to disable')
    parser.add_argument('--do_train', action='store_true',
                        help='run training')
    parser.add_argument('--do_test', action='store_true',
                        help='run testing')
    parser.add_argument('--make_hsmm_graphs',
                        action='store_true',
                        help='Generate graphs of hsmm parameters.')
    parser.add_argument('--model_to_test',
                        type=str,
                        default='',
                        help='model filename')
    return parser


def make_hsmm_parameter_graphs(
        model,
        model_filename,
        save_mode_embeddings=False,
        save_initial_probability=False,
        save_emission_quality_probability=False,
        save_marginal_emission_probability=False,
        save_residence_probability=False,
        save_transition_probability=False,
        save_transition_stationary=False,
        save_unigram_probability=False
):
    # make graph of model parameters
    model.eval()
    params = model.get_hsmm_params()
    marginal_emission_distribution = params['marginal_emission_chroma']

    if save_mode_embeddings:
        mode_embeddings = params['mode_emb'][:model.num_modes]
        num_cols = 2
        num_rows = int(math.ceil(model.num_modes / float(num_cols)))
        df_mode_emb = pd.DataFrame(mode_embeddings.tolist())
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=df_mode_emb.T,
            data_name='',
            prob_name='mode_emb',
            legend=False,
            figsize=(num_cols * 4, num_rows * 4),
            layout=(num_rows, num_cols),
            subplots=True
        )

    if save_initial_probability:
        num_cols = min(model.num_modes, 4)
        num_rows = int(math.ceil(model.num_modes / float(num_cols)))
        initial_root_distribution = params['initial_root'][:model.num_modes]
        df_initial_root = pd.DataFrame(initial_root_distribution.tolist())
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=df_initial_root.T,
            data_name='',
            prob_name='initial_root',
            legend=False,
            figsize=(num_cols * 4, num_rows * 4),
            layout=(num_rows, num_cols),
            subplots=True
        )

    if save_marginal_emission_probability:
        num_cols = 4
        num_rows = int(math.ceil((model.num_roots + 1) / 4.0))
        emission_index = ['s{}'.format(i) for i in range(model.num_roots)]
        emission_columns = ['{}'.format(i) for i in range(model.chroma_dim)]
        for mode in range(model.num_modes):
            df_emission = pd.DataFrame(
                marginal_emission_distribution[mode].tolist(), index=emission_index, columns=emission_columns)
            HmmVisualizer.save_bar_plot(
                model_filename=model_filename,
                df_data=df_emission.T,
                data_name='',
                prob_name='marginal_emission-k{}'.format(mode),
                legend=False,
                figsize=(num_cols * 4, num_rows * 4),
                layout=(num_rows, num_cols),
                subplots=True
            )

    if save_emission_quality_probability:
        emission_quality_probability = params['emission_quality'][:model.num_modes]  # (M, R+1, Q + 1)
        for mode in range(model.num_modes):
            num_rows = int(math.ceil((model.num_roots + 1) / 4.0))
            num_cols = 4
            emission_quality_index = ['s{}'.format(i) for i in range(model.num_roots + 1)]
            df_emission_quality = pd.DataFrame(emission_quality_probability[mode].tolist(), index=emission_quality_index)
            HmmVisualizer.save_bar_plot(
                model_filename=model_filename,
                df_data=df_emission_quality.T,
                data_name='',
                prob_name='emission-quality-k{}'.format(mode),
                figsize=(num_cols * 4, num_rows * 4),
                layout=(num_rows, num_cols),
                subplots=True)

    if save_transition_probability:
        root_transition_distribution = params['transition_root'][:model.num_modes]
        for mode in range(model.num_modes):
            num_cols = 4
            num_rows = int(math.ceil((model.num_roots + 1) / 4.0))
            root_transition_index = ['s{}'.format(i) for i in range(model.num_roots + 1)]
            df_transition_root = pd.DataFrame(root_transition_distribution[mode].tolist(), index=root_transition_index)
            HmmVisualizer.save_bar_plot(
                model_filename=model_filename,
                df_data=df_transition_root.T,
                data_name='',
                prob_name='root-transition-k{}'.format(mode),
                figsize=(num_cols * 4, num_rows * 4),
                layout=(num_rows, num_cols),
                subplots=True)

    if save_residence_probability:
        residence_distribution = params['residence']
        num_rows = max(int(math.ceil((model.num_roots + 1) / 4.0)), 2)
        num_cols = 4
        residence_index = [i for i in range(model.num_roots + 1)]
        df_residence = pd.DataFrame(residence_distribution.tolist(), index=residence_index)
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=df_residence.T,
            data_name='',
            prob_name='residence',
            figsize=(num_cols * 4, num_rows * 4),
            layout=(num_rows, num_cols),
            subplots=True
        )

    if save_transition_stationary:
        num_cols = model.num_modes
        num_rows = int(math.ceil(model.num_modes / float(num_cols)))
        df_stationary_transition = pd.DataFrame(params['transition_stationary'].tolist())
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=df_stationary_transition.T,
            data_name='',
            prob_name='stationary_transition',
            legend=False,
            figsize=(num_cols * 4, num_rows * 4),
            layout=(num_rows, num_cols),
            subplots=True
        )

    if save_unigram_probability:
        unigram_probability = params['unigram']
        num_cols = 2
        num_rows = int(math.ceil(model.num_modes / float(num_cols)))
        df_unigram = pd.DataFrame(unigram_probability.tolist())
        HmmVisualizer.save_bar_plot(
            model_filename=model_filename,
            df_data=df_unigram.T,
            data_name='',
            prob_name='unigram',
            legend=False,
            figsize=(num_cols * 4, num_rows * 4),
            layout=(num_rows, num_cols),
            subplots=True
        )


def event_to_segment(choral_id, event_items):
    event_length = len(event_items)
    seg_items = [[choral_id, 0, 0, event_items[0]]]
    for ie in range(1, event_length):
        if event_items[ie] == seg_items[-1][-1]:
            seg_items[-1][2] = ie
        else:
            seg_items.append([choral_id, ie, ie, event_items[ie]])
    return seg_items


def run_evaluate_bach60(config, model, instances, logger):
    model.eval()
    eval_output = Evaluator.evaluate(
        instances, model, config, logger, viterbi_output=True, metadata_output=True)
    params = model.get_hsmm_params()
    tonic_ids = params['tonic_ids']

    name_to_pc = {}
    for pc, names in enumerate(PITCH_NAME_LABELS):
        names = [n.replace(')', '') for n in names.split('(')]
        for name in names:
            name_to_pc[name] = pc

    dict_instances = {}
    for instance in instances:
        choral_id = instance[META_DATA]['choral_id']
        if choral_id not in dict_instances:
            dict_instances[choral_id] = {}
        dict_instances[choral_id] = instance

    dict_output = {}
    total_fullchord_accuracy = 0
    total_rootchord_accuracy = 0
    total_event_length = 0
    for be in eval_output['output']:
        for states, residences, qualities, obs, length, metadata in zip(
                be[STATES],
                be[RESIDENCES],
                be['qualities'],
                be['observation_chroma'],
                be['sequence_length'],
                be[META_DATA]
        ):
            choral_id = metadata['choral_id']
            raw_gold_labels = dict_instances[choral_id][META_DATA]['chord_label']
            gold_fullchord_labels = []
            gold_rootchord_labels = []
            for gl in raw_gold_labels:
                gold_name = gl[:2].replace('_', '')
                gold_quality = gl[2:]
                gold_pc = name_to_pc[gold_name]
                gold_fullchord_labels.append('{}_{}'.format(gold_pc, gold_quality))
                gold_rootchord_labels.append(gold_pc)

            key_ids = [int(s / 13) for s in states]
            root_pcs = [s % 13 for s in states]
            key_names = []
            pred_fullchord_labels = []
            pred_rootchord_labels = []
            for t, (ki, rpc, quality) in enumerate(zip(key_ids, root_pcs, qualities)):
                mode = int(ki / 12)
                key_shift = ki % 12
                key_pc = ((tonic_ids[mode] + key_shift) % 12).item()
                key_names.append(PITCH_NAME_LABELS[key_pc])
                # bach60 does not distinguish between dominant and major7.
                if quality == '7':
                    quality = 'M7'
                # bach60 represents chords in root pitch class, not in degrees.
                pred_fullchord_labels.append('{}_{}'.format(rpc, quality))
                pred_rootchord_labels.append(rpc)

            assert len(gold_fullchord_labels) == len(gold_rootchord_labels) == len(pred_fullchord_labels) == len(
                pred_rootchord_labels)
            piece_event_fullchord_accuracy = 0
            piece_event_rootchord_accuracy = 0
            for ie in range(len(gold_fullchord_labels)):
                if gold_fullchord_labels[ie] == pred_fullchord_labels[ie]:
                    piece_event_fullchord_accuracy += 1
                if gold_rootchord_labels[ie] == pred_rootchord_labels[ie]:
                    piece_event_rootchord_accuracy += 1
            total_fullchord_accuracy += piece_event_fullchord_accuracy
            total_rootchord_accuracy += piece_event_rootchord_accuracy
            total_event_length += len(gold_fullchord_labels)

            assert choral_id not in dict_output
            dict_output[choral_id] = {
                'observation_chroma': obs,
                'sequence_length': length,
                STATES: states,
                RESIDENCES: residences,
                'key_names': key_names,
                'root_pitch_classes': root_pcs,
                'pred_fullchord_labels': pred_fullchord_labels,
                'pred_rootchord_labels': pred_rootchord_labels,
                'gold_fullchord_labels': gold_fullchord_labels,
                'gold_rootchord_labels': gold_rootchord_labels,
                'qualities': qualities,
                META_DATA: metadata
            }
            dict_output[choral_id]['eval_metrics'] = {
                'fullchord_accuracy': piece_event_fullchord_accuracy / float(len(gold_fullchord_labels)),
                'rootchord_accuracy': piece_event_rootchord_accuracy / float(len(gold_rootchord_labels)),
            }

    dict_output['total_eval_metrics'] = {
        'fullchord_accuracy': total_fullchord_accuracy / float(total_event_length),
        'rootchord_accuracy': total_rootchord_accuracy / float(total_event_length),
    }
    logger.info('fullchord_accuracy: {}'.format(total_fullchord_accuracy / float(total_event_length)))
    logger.info('rootchord_accuracy: {}'.format(total_rootchord_accuracy / float(total_event_length)))
    return dict_output


def get_labeled_score(riemen, gold, pred, offsets):
    import music21 as m21
    from music21.corpus.chorales import ChoraleListRKBWV

    gold_dict = dict([(g[1], g) for g in gold])
    pred_dict = dict([(p[1], p) for p in pred])

    bwv = ChoraleListRKBWV().byRiemenschneider[riemen]['bwv']
    score = m21.corpus.parse('bwv{}'.format(bwv))
    measure_to_offset_id = {}
    cursol = 0
    seen_ids = []
    for mi, measure in enumerate(score.parts[-1].getElementsByClass('Measure')):
        for note in measure.notes:
            seen_ids.append(note.id)
        cursol_start = cursol
        for o in offsets[cursol_start:]:
            if (0 < mi) and o < measure.offset:
                if (mi - 1) not in measure_to_offset_id:
                    measure_to_offset_id[mi - 1] = []
                measure_to_offset_id[mi - 1].append(cursol)
                cursol += 1
    measure_to_offset_id[mi] = list(range(cursol, len(offsets)))

    prev_gold_key = None
    prev_pred_key = None
    for mi, measure in enumerate(score.parts[-1].getElementsByClass('Measure')):
        for offset in measure_to_offset_id[mi]:
            if offset in gold_dict:
                gold_key, gold_label = gold_dict[offset][-1]
                if prev_gold_key == gold_key:
                    gold_key_label = ''
                else:
                    gold_key_label = '{}: '.format(gold_key)
                    prev_gold_key = gold_key
                gold_label = gold_key_label + gold_label
            else:
                gold_label = ''
            if offset in pred_dict:
                pred_key, pred_label = pred_dict[offset][-1]
                if prev_pred_key == pred_key:
                    pred_key_label = ''
                else:
                    pred_key_label = '{}: '.format(pred_key)
                    prev_pred_key = pred_key
                pred_label = pred_key_label + pred_label
            else:
                pred_label = ''

            if bool(gold_label) or bool(pred_label):
                added = False
                for note in measure.notes:
                    if (measure.offset + note.offset) == offsets[offset]:
                        note.lyric = '{}\n{}'.format(pred_label, gold_label)
                        added = True
                        break
                if not added:
                    for other_part in score.parts[::-1][1:]:
                        if added:
                            break
                        # search other parts
                        for note in other_part.getElementsByClass('Measure')[mi]:
                            if (measure.offset + note.offset) == offsets[offset]:
                                note.lyric = '{}\n{}'.format(pred_label, gold_label)
                                added = True
                                break
    return score


def run_evaluate_bach_4part(config, model, instances, logger, output_dir):
    import json
    with open('human_annotation_music21/bach_chorale_annotation_m21.json', 'r') as f:
        bach_choral_annotation_m21 = json.load(f)
    output_dir = output_dir / Path('score_labeled')
    if not output_dir.is_dir():
        output_dir.mkdir()

    if config.pivot_chord_selection == 'first':
        sub_record_id = 0
    elif config.pivot_chord_selection == 'last':
        sub_record_id = -1
    else:
        raise NotImplementedError

    eval_output = Evaluator.evaluate(
        instances, model, config, logger, viterbi_output=True, metadata_output=True)
    params = model.get_hsmm_params()
    tonic_ids = params['tonic_ids']
    tonic_qualities = params['tonic_qualities']

    name_to_pc = {}
    for pc, names in enumerate(PITCH_NAME_LABELS):
        names = [n.replace(')', '') for n in names.split('(')]
        for name in names:
            name_to_pc[name] = pc

    dict_instances = {}
    for instance in instances:
        if 'choral_id' in instance[META_DATA]:
            choral_id = instance[META_DATA]['choral_id']
        elif 'riemenschneider' in instance[META_DATA]:
            choral_id = instance[META_DATA]['riemenschneider']
        else:
            raise NotImplementedError
        if choral_id not in dict_instances:
            dict_instances[choral_id] = {}
        if 'section_id' in instance[META_DATA]:
            section_id = int(instance[META_DATA]['section_id'])
        else:
            section_id = 0
        assert section_id not in dict_instances[choral_id]
        dict_instances[choral_id][section_id] = instance

    dict_golds = {}
    for choral_id in bach_choral_annotation_m21.keys():
        if int(choral_id) in dict_instances:
            min_duration = dict_instances[int(choral_id)][0][META_DATA]['min_duration']
            annotation = bach_choral_annotation_m21[choral_id]['annotation']
            ann_key_sig = bach_choral_annotation_m21[choral_id]['key_sigs']
            ann_time_sig = bach_choral_annotation_m21[choral_id]['time-signature']
            if ann_time_sig in ['3/4', '4/4']:
                beat_positions = [str(1.0 + b * min_duration) for b in range(int(float(ann_time_sig[0]) / min_duration))]
            else:
                raise NotImplementedError
            prev_record = None
            gold_key_names = []
            gold_fullchord_labels = []
            gold_rootchord_labels = []
            gold_qualities = []
            gold_metrics = []
            metric = 0.0
            for measure in annotation.keys():
                if measure == '0':
                    start_beat = list(annotation['0'].keys())[0]
                    start_beat_id = beat_positions.index(start_beat)
                else:
                    start_beat_id = 0
                for beat in beat_positions[start_beat_id:]:
                    if beat in annotation[measure].keys():
                        prev_record = annotation[measure][beat][sub_record_id]
                    gold_metrics.append(metric)
                    gold_key_names.append(prev_record[0].replace('-', 'b'))
                    fullchord_label = prev_record[1].split('[')[0]  # remove comment
                    # Dealing with mixed vii/o7 in viio7 in annotations
                    fullchord_label = fullchord_label.replace('/o', 'o')
                    gold_fullchord_labels.append(fullchord_label)
                    fcll = prev_record[1].lower()
                    if fcll.startswith('iii') or fcll.startswith('vii'):
                        rc, quality = fullchord_label[:3], fullchord_label[3:]
                    elif fcll.startswith('ii') or fcll.startswith('iv') or fcll.startswith('vi'):
                        rc, quality = fullchord_label[:2], fullchord_label[2:]
                    elif fcll.startswith('v') or fcll.startswith('i'):
                        rc, quality = fullchord_label[:1], fullchord_label[1:]
                    elif fcll.startswith('bvii'):
                        # The only exception: bvii7[maj7]
                        rc, quality = fullchord_label[:4], fullchord_label[4:]
                    else:
                        raise NotImplementedError
                    # slash root
                    quality_splits = quality.split('/')
                    if ('/' in quality) and (not quality_splits[-1].isdecimal()):
                        slash_root = '/{}'.format(quality[-(len(quality_splits[-1])):])
                        rc += slash_root
                        quality = quality[:-(len(quality_splits[-1]) + 1)]
                    gold_rootchord_labels.append(rc)
                    gold_qualities.append(quality)
                    metric += min_duration
            assert int(choral_id) not in dict_golds
            dict_golds[int(choral_id)] = {
                'key_names': gold_key_names,
                'fullchord_labels': gold_fullchord_labels,
                'rootchord_labels': gold_rootchord_labels,
                'qualities': gold_qualities,
                'metrics': gold_metrics,
                'key_sigs': ann_key_sig,
                'time_sig': ann_time_sig
            }

    dict_section_output = {}
    for be in eval_output['output']:
        for states, residences, qualities, obs, length, metadata in zip(
                be[STATES],
                be[RESIDENCES],
                be['qualities'],
                be['observation_chroma'],
                be['sequence_length'],
                be[META_DATA]
        ):
            choral_id = metadata['riemenschneider']
            section_id = int(metadata['section_id'])
            bass_pcs = metadata['bass']
            key_sigs = [ks.name for ks in metadata['key_sigs']]
            metadata['key_sigs'] = key_sigs

            key_ids = [int(s / 13) for s in states]
            root_pcs = [s % 13 for s in states]
            key_names = []
            pred_fullchord_labels = []
            pred_rootchord_labels = []
            for t, (ki, rpc, quality) in enumerate(zip(key_ids, root_pcs, qualities)):
                mode = int(ki / 12)
                key_shift = ki % 12
                key_pc = ((tonic_ids[mode] + key_shift) % 12).item()
                key_name = PITCH_NAME_LABELS_PRIOR_DICT[tonic_qualities[mode]][key_pc]
                if '(' in key_name and bool(key_sigs):
                    if '#' in key_sigs[0]:
                        key_name = key_name.split('(')[0]
                    else:
                        assert '-' in key_sigs[0], key_sigs
                        key_name = key_name.split('(')[1][:-1]
                if tonic_qualities[mode] in ['m', 'd']:
                    key_name = key_name.lower()
                key_names.append(key_name)
                root_relative_pc = (rpc - key_pc) % 12
                bass_relative_pc = (bass_pcs[t] - key_pc) % 12
                sub_root_bass = (root_relative_pc - bass_relative_pc) % 12
                degree = PITCH_TO_DEGREE[root_relative_pc]
                if quality == 'Rest':
                    pred_fullchord_labels.append('Rest')
                    pred_rootchord_labels.append('Rest')
                else:
                    if quality == '7' or 'M' in quality:
                        degree = degree.upper()
                    sub_quality = 'o' if 'd' in quality else None
                    is_seventh = True if '7' in quality else False
                    if sub_root_bass in INVERSION_DICT[quality]:
                        quality = INVERSION_DICT[quality][sub_root_bass]
                        if sub_quality is not None:
                            quality = sub_quality + quality
                        pred_fullchord_labels.append('{}{}'.format(degree, quality))
                    elif is_seventh:  # seventh base chord
                        quality = '7'
                        if sub_quality is not None:
                            quality = sub_quality + quality
                        pred_fullchord_labels.append('{}{}'.format(degree, quality))
                    else:
                        quality = None
                        pred_fullchord_labels.append(degree)
                    pred_rootchord_labels.append(degree)

            if choral_id not in dict_section_output:
                dict_section_output[choral_id] = {}
            assert section_id not in dict_section_output[choral_id]
            dict_section_output[choral_id][section_id] = {
                'observation_chroma': obs,
                'sequence_length': length,
                STATES: states,
                RESIDENCES: residences,
                'key_names': key_names,
                'root_pitch_classes': root_pcs,
                'pred_fullchord_labels': pred_fullchord_labels,
                'pred_rootchord_labels': pred_rootchord_labels,
                'qualities': qualities,
                META_DATA: metadata
            }

    test_riemens = sorted(list(dict_golds.keys()))
    dict_output = {}
    total_event_key_accuracy = 0
    total_event_fullchord_accuracy = 0
    total_event_rootchord_accuracy = 0
    total_event_fullanalysis_accuracy = 0

    total_event_length = 0
    for riemen in test_riemens:
        # gather sections
        assert riemen not in dict_output
        dict_output[riemen] = {}
        pred_fullchord_labels = []
        pred_rootchord_labels = []
        pred_fullanalysis_labels = []
        pred_key_names = []
        pred_metrics = []
        obs_pitches = []
        for section_id in sorted(list(dict_section_output[riemen].keys())):
            section_item = dict_section_output[riemen][section_id]
            pred_fullchord_labels.extend(section_item['pred_fullchord_labels'])
            pred_rootchord_labels.extend(section_item['pred_rootchord_labels'])
            pred_fullanalysis_labels.extend([(k, fc) for k, fc in zip(section_item['key_names'], section_item['pred_fullchord_labels'])])
            pred_key_names.extend(section_item['key_names'])
            pred_metrics.extend(section_item[META_DATA]['metrics'])
            section_length = len(section_item[META_DATA]['metrics'])
            for obs in section_item['observation_chroma'].tolist()[:section_length]:
                obs_pitches.append(tuple([io for (io, o) in enumerate(obs) if 0 < o]))
        dict_output[riemen]['key_names'] = pred_key_names
        dict_output[riemen]['fullchord_labels'] = pred_fullchord_labels
        dict_output[riemen]['rootchord_labels'] = pred_rootchord_labels
        dict_output[riemen]['metrics'] = pred_metrics
        dict_output[riemen]['observation_pitchClasses'] = obs_pitches
        # The parsing of gold_data does not take into account the length of the last measure
        # (which may be shorter in the case of an auftakt), so it may be a little longer than pred.
        assert len(pred_metrics) <= len(dict_golds[riemen]['metrics']), (len(dict_golds[riemen]['metrics']), len(pred_metrics))
        for k in ['metrics', 'key_names', 'fullchord_labels', 'rootchord_labels', 'qualities']:
            dict_golds[riemen][k] = dict_golds[riemen][k][:len(pred_metrics)]
        dict_golds[riemen]['fullanalysis_labels'] = [(k, fc) for k, fc in zip(dict_golds[riemen]['key_names'], dict_golds[riemen]['fullchord_labels'])]
        event_length = len(pred_metrics)

        if set(dict_golds[riemen]['key_sigs']) == set(dict_section_output[riemen][0][META_DATA]['key_sigs']):
            # event level accuracy
            piece_event_key_accuracy = 0
            piece_event_fullchord_accuracy = 0
            piece_event_rootchord_accuracy = 0
            piece_event_fullanalysis_accuracy = 0
            for ie in range(event_length):
                if dict_golds[riemen]['key_names'][ie] == pred_key_names[ie]:
                    piece_event_key_accuracy += 1
                if dict_golds[riemen]['fullchord_labels'][ie] == pred_fullchord_labels[ie]:
                    piece_event_fullchord_accuracy += 1
                if dict_golds[riemen]['rootchord_labels'][ie] == pred_rootchord_labels[ie]:
                    piece_event_rootchord_accuracy += 1
                if (dict_golds[riemen]['key_names'][ie] == pred_key_names[ie]) and (dict_golds[riemen]['fullchord_labels'][ie] == pred_fullchord_labels[ie]):
                    piece_event_fullanalysis_accuracy += 1

            total_event_length += event_length
            total_event_key_accuracy += piece_event_key_accuracy
            total_event_fullchord_accuracy += piece_event_fullchord_accuracy
            total_event_rootchord_accuracy += piece_event_rootchord_accuracy
            total_event_fullanalysis_accuracy += piece_event_fullanalysis_accuracy

            gold_seg_fullanalysis_labels = event_to_segment(riemen, dict_golds[riemen]['fullanalysis_labels'])
            pred_seg_fullanalysis_labels = event_to_segment(riemen, pred_fullanalysis_labels)

            dict_output[riemen]['eval_metrics'] = {
                'key_accuracy': piece_event_key_accuracy / float(event_length),
                'fullchord_accuracy': piece_event_fullchord_accuracy / float(event_length),
                'rootchord_accuracy': piece_event_rootchord_accuracy / float(event_length),
                'fullanalysis_accuracy': piece_event_fullanalysis_accuracy / float(event_length),
            }

            labeled_score = get_labeled_score(
                riemen,
                gold=gold_seg_fullanalysis_labels,
                pred=pred_seg_fullanalysis_labels,
                offsets=pred_metrics
            )
            labeled_score.write('musicxml', str(output_dir / Path('riemenschneider{}-seed{}.mxl'.format(riemen, config.seed))))

            # add gold
            dict_output[riemen]['gold'] = dict_golds[riemen]

    total_event_key_accuracy /= float(total_event_length)
    total_event_fullchord_accuracy /= float(total_event_length)
    total_event_rootchord_accuracy /= float(total_event_length)
    total_event_fullanalysis_accuracy /= float(total_event_length)

    dict_output['total_eval_metrics'] = {
        'key_accuracy': total_event_key_accuracy,
        'fullchord_accuracy': total_event_fullchord_accuracy,
        'rootchord_accuracy': total_event_rootchord_accuracy,
        'fullanalysis_accuracy': total_event_fullanalysis_accuracy,
    }
    logger.info('key_accuracy: {}'.format(total_event_key_accuracy))
    logger.info('fullchord_accuracy: {}'.format(total_event_fullchord_accuracy))
    logger.info('rootchord_accuracy: {}'.format(total_event_rootchord_accuracy))
    return dict_output


def run(args):
    set_seed(args.seed, args.device)

    config = Config(args)
    assert not (args.do_train and args.do_test)
    assert not (args.do_train and args.make_hsmm_graphs)

    if args.do_test:
        assert config.key_preprocessing == KEY_PREPROCESS_NONE
        dest_log = Path(args.model_to_test).parents[0] / Path(TEST) / Path(
            '{}-{}.log'.format(Path(args.model_to_test).stem, TEST))
    elif args.make_hsmm_graphs:
        dest_log = Path(args.model_to_test).parents[0] / Path('hsmm_graphs') / Path(
            '{}-{}.log'.format(Path(args.model_to_test).stem, 'hsmm_graphs'))
    else:
        dest_log = config.dir_output / Path(config.log_filename)

    logger = create_logger(dest_log=dest_log)
    logger.info('\n'.join(['{}: {}'.format(k, v) for k, v in vars(args).items()]))

    if args.do_test or args.make_hsmm_graphs:
        data_dir = Path(args.model_to_test).parents[1]
    else:
        data_dir = config.dir_instance_output

    instance_path = data_dir / Path('{}.pkl'.format(config.data_suffix))
    if 'bach60' in config.dataset:
        reader = Bach60Reader(config)
        if instance_path.is_file():
            instances = pickle.load(instance_path.open('rb'))
        else:
            instances = reader.create_instance(logger)
            pickle.dump(instances, instance_path.open('wb'))
    elif 'bach-4part' in config.dataset:
        reader = BachChoraleReader(config)
        if instance_path.is_file():
            instances = pickle.load(instance_path.open('rb'))
        else:
            instances = reader.create_instance(logger)
            pickle.dump(instances, instance_path.open('wb'))
    else:
        raise NotImplementedError

    train_instances, dev_instances, test_instances = reader.split_train_dev_test(instances, logger)

    if args.do_train:
        if config.model_to_initialize is not None:
            pretrained_checkpoint = torch.load(args.model_to_initialize, map_location=torch.device(args.device))
            model = NeuralHSMM(config)
            model.load_state_dict(pretrained_checkpoint['model'], strict=False)
        else:
            model = NeuralHSMM(config)
        logger.info('Start training')
        model.to(args.device)

        best_model, model_filename = Trainer.train(
            train_instances, dev_instances, model, config, logger)

    if args.make_hsmm_graphs:
        checkpoint = torch.load(args.model_to_test, map_location=torch.device(args.device))

        model = NeuralHSMM(config)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()

        model_filename = Path(args.model_to_test)
        output_dir = model_filename.parent / Path('hsmm_graphs')

        if not output_dir.is_dir():
            logger.info('create {}'.format(output_dir.name))
            output_dir.mkdir()

        make_hsmm_parameter_graphs(
            model,
            model_filename=output_dir / Path(model_filename).stem,
            save_mode_embeddings=True,
            save_initial_probability=True,
            save_emission_quality_probability=True,
            save_marginal_emission_probability=True,
            save_residence_probability=True,
            save_transition_probability=True,
            save_transition_stationary=True,
            save_unigram_probability=True
        )

    if args.do_test:
        instance_label = TEST
        instance_for_test = test_instances

        checkpoint = torch.load(args.model_to_test, map_location=torch.device(args.device))

        model = NeuralHSMM(config)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()

        model_filename = Path(args.model_to_test)
        output_dir = model_filename.parent / Path(instance_label)

        if not output_dir.is_dir():
            logger.info('create {}'.format(output_dir.name))
            output_dir.mkdir()

        if config.dataset == 'bach60':
            run_evaluate_bach60(config, model, instance_for_test, logger)
        elif 'bach-4part-mxl' in config.dataset:
            run_evaluate_bach_4part(config, model, instance_for_test, logger, output_dir)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    _args = add_arguments().parse_args()
    run(_args)


