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
from core.preprocess.reader import MxlReader, KEY_DE2EN
from core.trainer.trainer import Trainer
from core.util.config import Config
from core.util.logging import create_logger
from core.util.util import set_seed

EVAL_RESOLUTION = 0.25  # 16th (Excluding bach60 dataset, which has already been pre-processed.)


def add_arguments():
    parser = argparse.ArgumentParser(prog='run')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='`cuda:n` where `n` is an integer, or `cpu`')
    parser.add_argument('--dataset',
                        type=str,
                        default='bach-4part-mxl',
                        choices=['bach60', 'bach-4part-mxl'],
                        help='Dataset name')
    parser.add_argument('--include_bracketed_annotations',
                        action='store_true',
                        help='Whether to include bracketed chords in the annotation.')
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
    parser.add_argument('--resolution_str',
                        type=str,
                        default='half-beat',
                        choices=['16th', '8th', 'half-beat', 'beat'],
                        help='Time resolution for predicting harmonic analysis')
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
    parser.add_argument('--dynamic_num_modes',
                        action='store_true',
                        help='Num_modes is not fixed and is determined by learning.')
    parser.add_argument('--acceptance_th',
                        type=float,
                        default=1e-2,
                        help='Threshold to increase number of modes')
    parser.add_argument('--cossim_limit',
                        type=float,
                        default=0.8,
                        help='Upper bound on the cosine similarity between a new mode and an already accepted mode.')
    parser.add_argument('--warmup_num_modes',
                        type=int,
                        default=16,
                        help='Number of warm-up epochs for num_modes optimisation.')
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


def get_labeled_score(config, filename, gold, raw_pred, chordlabel_key='fullchord'):
    import music21 as m21
    from music21.stream import Voice
    from music21.note import Note, Rest
    from music21.chord import Chord
    if config.dataset in ['bach-4part-mxl']:
        bwv = filename.split('bwv')[-1]
        score = m21.corpus.parse('bwv{}'.format(bwv))
    else:
        raise NotImplementedError

    # Remove successions of the same label
    raw_pred = sorted(raw_pred.items(), key=lambda x:x[0])
    pred = {raw_pred[0][0]: raw_pred[0][1]}
    for i in range(1, len(raw_pred)):
        prev_p = raw_pred[i - 1][1]
        cur_p = raw_pred[i][1]
        if 'fullchord' in prev_p and prev_p['fullchord'] is not None:
            if not ((prev_p['key_name'] == cur_p['key_name']) and (prev_p['fullchord'] == cur_p['fullchord'])):
                pred[cur_p['time']] = cur_p
        else:
            if not ((prev_p['key_name'] == cur_p['key_name']) and (prev_p['rootchord'] == cur_p['rootchord'])):
                pred[cur_p['time']] = cur_p

    prev_gold_key = None
    prev_pred_key = None
    prev_gold_pi = None
    prev_pred_pi = None
    gold_remaining = list(gold.keys())
    pred_remaining = list(pred.keys())
    for pi in range(len(score.parts))[::-1]:
        # bass first
        for measure in score.parts[pi].getElementsByClass('Measure'):
            m_iter = [v for v in measure.getElementsByClass(Voice)]
            if bool(measure.elements):
                m_iter += [measure]
            for voice in m_iter:
                for note in [vn for vn in voice.elements if isinstance(vn, Note) or isinstance(vn, Rest) or isinstance(vn, Chord)]:
                    # gold
                    gold_label = ''
                    if voice.offset + note.offset in gold_remaining:
                        if (prev_gold_pi is None) or prev_gold_pi != pi:
                            gold_label += 'gold:'
                            prev_gold_pi = pi
                        if (prev_gold_key is None) or prev_gold_key != gold[voice.offset + note.offset]['key_name']:
                            gold_label += '{}:'.format(gold[voice.offset + note.offset]['key_name'])
                            prev_gold_key = gold[voice.offset + note.offset]['key_name']
                        gold_label += gold[voice.offset + note.offset][chordlabel_key]
                        gold_remaining.remove(voice.offset + note.offset)
                    # pred
                    pred_label = ''
                    if voice.offset + note.offset in pred_remaining:
                        if (prev_pred_pi is None) or prev_pred_pi != pi:
                            pred_label += 'pred:'
                            prev_pred_pi = pi
                        if (prev_pred_key is None) or prev_pred_key != pred[voice.offset + note.offset]['key_name']:
                            pred_label += '{}:'.format(pred[voice.offset + note.offset]['key_name'])
                            prev_pred_key = pred[voice.offset + note.offset]['key_name']
                        pred_label += pred[voice.offset + note.offset][chordlabel_key]
                        pred_remaining.remove(voice.offset + note.offset)
                    label = '{}\n{}'.format(gold_label, pred_label)
                    if label.strip():
                        note.lyric = label
        if not(bool(gold_remaining) or bool(pred_remaining)):
            break
    return score


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
        assert choral_id not in dict_instances
        dict_instances[choral_id] = instance

    dict_output = {}
    total_fullchord_accuracy = 0
    total_rootchord_accuracy = 0
    total_event_length = 0
    for be in eval_output['output']:
        for states, residences, qualities, metadata in zip(
                be[STATES],
                be[RESIDENCES],
                be['qualities'],
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


def run_evaluate(config, model, model_filename, instances, logger, output_dir):
    output_dir = output_dir / Path('score_labeled')
    if not output_dir.is_dir():
        output_dir.mkdir()
    if config.pivot_chord_selection == 'first':
        sub_record_id = 0
    elif config.pivot_chord_selection == 'last':
        sub_record_id = -1
    else:
        raise NotImplementedError

    name_to_pc = {}
    for pc, names in enumerate(PITCH_NAME_LABELS):
        names = [n.replace(')', '') for n in names.split('(')]
        for name in names:
            name_to_pc[name] = pc

    if 'bach-4part' in config.dataset:
        import json
        with open('human_annotation_music21/bach_chorale_annotation_m21.json', 'r') as f:
            bach_choral_annotation_m21 = json.load(f)

    eval_output = Evaluator.evaluate(
        instances, model, config, logger, viterbi_output=True, metadata_output=True)
    params = model.get_hsmm_params()
    tonic_ids = params['tonic_ids']
    tonic_qualities = params['tonic_qualities']

    # gather instances
    if 'seg_index' in instances[0][META_DATA]:
        temp_gathered_instances = {}
        for instance in instances:
            fn = instance[META_DATA]['filename']
            num_segments = instance[META_DATA]['num_segments']
            timestep = instance[META_DATA]['timestep']
            seg_index = instance[META_DATA]['seg_index']
            metadata = dict([(k, v) for k, v in instance[META_DATA].items() if k not in ['seg_index', 'timestep']])
            if fn not in temp_gathered_instances:
                temp_gathered_instances[fn] = {
                    META_DATA: metadata,
                    'timestep': [None] * num_segments,
                    'sequence_length': 0,
                    'x_chroma': [None] * num_segments,
                    'x_bass': [None] * num_segments
                }
            temp_gathered_instances[fn]['timestep'][seg_index] = timestep
            temp_gathered_instances[fn]['sequence_length'] += instance['sequence_length'].to_tensor().item()
            temp_gathered_instances[fn]['x_chroma'][seg_index] = instance['x_chroma'].to_tensor().tolist()
            temp_gathered_instances[fn]['x_bass'][seg_index] = instance['x_bass'].to_tensor().tolist()

        gathered_instances = {}
        for fn in temp_gathered_instances:
            timestep = []
            x_chroma = []
            x_bass = []
            for seg_index in range(temp_gathered_instances[fn][META_DATA]['num_segments']):
                timestep.extend(temp_gathered_instances[fn]['timestep'][seg_index])
                x_chroma.extend(temp_gathered_instances[fn]['x_chroma'][seg_index])
                x_bass.extend(temp_gathered_instances[fn]['x_bass'][seg_index])
            gathered_instances[fn] = {
                META_DATA: temp_gathered_instances[fn][META_DATA],
                'timestep': timestep,
                'sequence_length': temp_gathered_instances[fn]['sequence_length'],
                'x_chroma': x_chroma,
                'x_bass': x_bass
            }
    else:
        gathered_instances = {}
        for instance in instances:
            gathered_instances[instance[META_DATA]['filename']] = {
                META_DATA: instance[META_DATA],
                'timestep': instance[META_DATA]['timestep'].copy(),
                'sequence_length': instance['sequence_length'].to_tensor().item(),
                'x_chroma': instance['x_chroma'].to_tensor().tolist(),
                'x_bass': instance['x_bass'].to_tensor().tolist()
            }

    # gather annotations
    golds = {}
    for instance in gathered_instances.values():
        fn = instance[META_DATA]['filename']
        # parse annotation
        ann = {}
        prev_offset = None
        if config.dataset == 'bach-4part-mxl':
            riemen = fn.split('-')[0][len('riemen'):]
            if not (set(bach_choral_annotation_m21[riemen]['key_sigs']) == set(
                [_ for _ in instance[META_DATA]['key_sigs'].split(',') if bool(_)])):
                logger.info('{}: Skip evaluation due to inconsistent key signatures. annotation:{}, source:{}'.format(
                    riemen,
                    set(bach_choral_annotation_m21[riemen]['key_sigs']),
                    set(instance[META_DATA]['key_sigs'].split(','))))
                continue
            measure2offset = {}
            for v in instance[META_DATA]['offset2mn'].values():
                if v['m_number'] in measure2offset:
                    assert (0.0 < v['m_paddingLeft']) or (v['m_offset'] == measure2offset[v['m_number']]['m_offset']), (
                    measure2offset, v)
                else:
                    measure2offset[v['m_number']] = {'m_offset': v['m_offset'], 'm_paddingLeft': v['m_paddingLeft']}
            for measure, m_ann in bach_choral_annotation_m21[riemen]['annotation'].items():
                measure = int(measure)
                for beat, mb_ann in m_ann.items():
                    offset = measure2offset[measure]['m_offset'] + (
                                float(beat) - measure2offset[measure]['m_paddingLeft'] - 1.0)
                    if bool(ann):
                        assert prev_offset < offset, (measure, beat, mb_ann)
                    key_name, fullchord_label = mb_ann[sub_record_id]
                    key_name = key_name.replace('-', 'b')
                    fullchord_label = fullchord_label.split('[')[0]  # remove comment
                    # Dealing with mixed vii/o7 in viio7 in annotations
                    fullchord_label = fullchord_label.replace('/o', 'o')
                    fcll = fullchord_label.lower()
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
                    ann[offset] = {
                        'time': offset,
                        'measure': measure,
                        'beat': beat,
                        'key_name': key_name,
                        'fullchord': fullchord_label,
                        'rootchord': rc,
                        'quality': quality
                    }
                    prev_offset = offset
        else:
            raise NotImplementedError
        assert fn not in golds
        golds[fn] = ann

    # gather outputs
    if 'seg_index' in eval_output['output'][0][META_DATA][0]:
        temp_gathered_outputs = {}
        for be in eval_output['output']:
            for states, residences, qualities, raw_metadata in zip(
                    be[STATES],
                    be[RESIDENCES],
                    be['qualities'],
                    be[META_DATA]
            ):
                fn = raw_metadata['filename']
                num_segments = raw_metadata['num_segments']
                timestep = raw_metadata['timestep']
                seg_index = raw_metadata['seg_index']
                metadata = dict([(k, v) for k, v in raw_metadata.items() if k not in ['seg_index', 'timestep']])
                if fn not in temp_gathered_outputs:
                    temp_gathered_outputs[fn] = {
                        META_DATA: metadata,
                        'timestep': [None] * num_segments,
                        'states': [None] * num_segments,
                        'residences': [None] * num_segments,
                        'qualities': [None] * num_segments
                    }
                temp_gathered_outputs[fn]['timestep'][seg_index] = timestep
                temp_gathered_outputs[fn]['states'][seg_index] = states
                temp_gathered_outputs[fn]['residences'][seg_index] = residences
                temp_gathered_outputs[fn]['qualities'][seg_index] = qualities
        gathered_outputs = {}
        for fn in temp_gathered_outputs:
            timestep = []
            states = []
            residences = []
            qualities = []
            for seg_index in range(temp_gathered_outputs[fn][META_DATA]['num_segments']):
                timestep.extend(temp_gathered_outputs[fn]['timestep'][seg_index])
                states.extend(temp_gathered_outputs[fn]['states'][seg_index])
                residences.extend(temp_gathered_outputs[fn]['residences'][seg_index])
                qualities.extend(temp_gathered_outputs[fn]['qualities'][seg_index])
            gathered_outputs[fn] = {
                META_DATA: temp_gathered_outputs[fn][META_DATA],
                'timestep': timestep,
                'states': states,
                'residences': residences,
                'qualities': qualities
            }
    else:
        gathered_outputs = {}
        for be in eval_output['output']:
            for states, residences, qualities, raw_metadata in zip(
                    be[STATES],
                    be[RESIDENCES],
                    be['qualities'],
                    be[META_DATA]
            ):
                metadata = dict([(k, v) for k, v in raw_metadata.items() if k not in ['seg_index', 'timestep']])
                gathered_outputs[metadata['filename']] = {
                    META_DATA: metadata,
                    'states': states,
                    'residences': residences,
                    'qualities': qualities
                }

    # gather prediction
    predictions = {}
    for fn, output in gathered_outputs.items():
        assert fn not in predictions
        timestep = gathered_instances[fn]['timestep']
        states = output['states']
        qualities = output['qualities']
        bass_midis = gathered_instances[fn]['x_bass']
        assert (gathered_instances[fn]['sequence_length'] ==
                len(gathered_instances[fn]['x_chroma']) ==
                len(timestep) ==
                len(states) ==
                len(bass_midis) ==
                len(qualities))
        predictions[fn] = {}
        key_ids = [int(s / 13) for s in states]
        root_pcs = [s % 13 for s in states]
        bass_pcs = [int(bm) % 12 if 0 <= bm else REST_INDEX for bm in bass_midis]
        for t, ki, rpc, quality, bass in zip(timestep, key_ids, root_pcs, qualities, bass_pcs):
            assert t not in predictions[fn]
            mode = int(ki / 12)
            key_shift = ki % 12
            key_pc = ((tonic_ids[mode] + key_shift) % 12).item()
            key_name = PITCH_NAME_LABELS_PRIOR_DICT[tonic_qualities[mode]][key_pc]
            if tonic_qualities[mode] in ['m', 'd']:
                key_name = key_name.lower()
            if quality == 'Rest':
                predictions[fn][t] = {
                    'time': t,
                    'key_name': key_name,
                    'fullchord': 'Rest',
                    'rootchord': 'Rest',
                    'quality': quality
                }
            else:
                root_relative_pc = (rpc - key_pc) % 12
                bass_relative_pc = (bass - key_pc) % 12
                sub_root_bass = (root_relative_pc - bass_relative_pc) % 12
                degree = PITCH_TO_DEGREE[root_relative_pc].lower()
                if quality == '7' or 'M' in quality:
                    degree = degree.upper()
                sub_quality = 'o' if 'd' in quality else None
                is_seventh = True if '7' in quality else False
                if (bass != REST_INDEX) and (sub_root_bass in INVERSION_DICT[quality]):
                    quality = INVERSION_DICT[quality][sub_root_bass]
                    if sub_quality is not None:
                        quality = sub_quality + quality
                elif is_seventh:  # seventh base chord
                    quality = '7'
                    if sub_quality is not None:
                        quality = sub_quality + quality
                else:
                    quality = ''
                fullchord = '{}{}'.format(degree, quality)
                predictions[fn][t] = {
                    'time': t,
                    'key_name': key_name,
                    'fullchord': fullchord,
                    'rootchord': degree,
                    'quality': quality
                }

    # accuracy
    accuracy_scores = {}
    total_event_length = 0
    total_key_accuracy = 0
    total_fullchord_accuracy = 0
    total_rootchord_accuracy = 0
    total_fullanalysis_accuracy = 0
    total_rootanaylsis_accuracy = 0
    for fn in golds.keys():
        gold = list(golds[fn].values())
        pred = list(predictions[fn].values())
        timestep = gathered_instances[fn]['timestep']
        resolution = gathered_instances[fn][META_DATA]['resolution']
        end_time = timestep[-1] + resolution
        eval_timestep_length = int((end_time / EVAL_RESOLUTION) + 0.5)
        gi = -1
        pi = -1
        piece_event_length = 0
        piece_key_accuracy = 0
        piece_fullchord_accuracy = 0
        piece_rootchord_accuracy = 0
        piece_fullanalysis_accuracy = 0
        piece_rootanaylsis_accuracy = 0
        for it in range(eval_timestep_length):
            t = it * EVAL_RESOLUTION
            if (gi < len(gold) - 1) and (gold[gi + 1]['time'] <= t):
                gi += 1
            if (pi < len(pred) - 1) and (pred[pi + 1]['time'] <= t):
                pi += 1
            assert 0 <= pi
            if 0 <= gi:
                # gold annotation may not start at time 0.0
                piece_event_length += 1
                if gold[gi]['key_name'] == pred[pi]['key_name']:
                    piece_key_accuracy += 1
                if gold[gi]['rootchord'] == pred[pi]['rootchord']:
                    piece_rootchord_accuracy += 1
                if (gold[gi]['key_name'] == pred[pi]['key_name']) and (gold[gi]['rootchord'] == pred[pi]['rootchord']):
                    piece_rootanaylsis_accuracy += 1
                if config.dataset in ['bach-4part-mxl']:
                    if gold[gi]['fullchord'] == pred[pi]['fullchord']:
                        piece_fullchord_accuracy += 1
                    if (gold[gi]['key_name'] == pred[pi]['key_name']) and (gold[gi]['fullchord'] == pred[pi]['fullchord']):
                        piece_fullanalysis_accuracy += 1
        total_event_length += piece_event_length
        total_key_accuracy += piece_key_accuracy
        total_fullchord_accuracy += piece_fullchord_accuracy
        total_rootchord_accuracy += piece_rootchord_accuracy
        total_fullanalysis_accuracy += piece_fullanalysis_accuracy
        total_rootanaylsis_accuracy += piece_rootanaylsis_accuracy
        # piece accuracy
        accuracy_scores[fn] = {
            'key_accuracy': piece_key_accuracy / float(piece_event_length),
            'rootchord_accuracy': piece_rootchord_accuracy / float(piece_event_length),
            'rootanalysis_accuracy': piece_rootanaylsis_accuracy / float(piece_event_length)
        }
        logger.info('--- {} ---'.format(fn))
        logger.info('key_accuracy: {}'.format(accuracy_scores[fn]['key_accuracy']))
        logger.info('rootchord_accuracy: {}'.format(accuracy_scores[fn]['rootchord_accuracy']))
        logger.info('rootanalysis_accuracy: {}'.format(accuracy_scores[fn]['rootanalysis_accuracy']))
        if config.dataset in ['bach-4part-mxl']:
            accuracy_scores[fn]['fullchord_accuracy'] = piece_fullchord_accuracy / float(piece_event_length)
            accuracy_scores[fn]['fullanalysis_accuracy'] = piece_fullanalysis_accuracy / float(piece_event_length)
            logger.info('fullchord_accuracy: {}'.format(accuracy_scores[fn]['fullchord_accuracy']))
            logger.info('fullanalysis_accuracy: {}'.format(accuracy_scores[fn]['fullanalysis_accuracy']))
        # score
        if config.dataset in ['bach-4part-mxl']:
            labeled_score = get_labeled_score(config, fn, golds[fn], predictions[fn], chordlabel_key='fullchord')
        else:
            raise NotImplementedError
        labeled_score.write(
            'musicxml',
            str(output_dir / Path('{}-{}.mxl'.format(Path(fn).stem, model_filename))))

    total_key_accuracy /= float(total_event_length)
    total_fullchord_accuracy /= float(total_event_length)
    total_rootchord_accuracy /= float(total_event_length)
    total_fullanalysis_accuracy /= float(total_event_length)
    total_rootanaylsis_accuracy /= float(total_event_length)

    accuracy_scores['total'] = {
        'key_accuracy_micro': total_key_accuracy,
        'rootchord_accuracy_micro': total_rootchord_accuracy,
        'rootanalysis_accuracy_micro': total_rootanaylsis_accuracy
    }
    logger.info('--- summary ---')
    logger.info('num_modes: {}'.format(model.num_modes))
    logger.info('key_accuracy: {}'.format(total_key_accuracy))
    logger.info('rootchord_accuracy: {}'.format(total_rootchord_accuracy))
    logger.info('rootanalysis_accuracy: {}'.format(total_rootanaylsis_accuracy))
    if config.dataset in ['bach-4part-mxl']:
        accuracy_scores['total']['fullchord_accuracy_micro'] = total_fullchord_accuracy
        accuracy_scores['total']['fullanalysis_accuracy_micro'] = total_fullanalysis_accuracy
        logger.info('fullchord_accuracy: {}'.format(total_fullchord_accuracy))
        logger.info('fullanalysis_accuracy: {}'.format(total_fullanalysis_accuracy))
    return accuracy_scores


def run(args):
    set_seed(args.seed, args.device)

    config = Config(args)
    if config.dataset == 'bach60':
        config.dir_dataset = 'bach60'
    else:
        config.dir_dataset = None
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
    if config.dataset == 'bach60':
        reader = Bach60Reader(config)
    else:
        reader = MxlReader(config)
    if instance_path.is_file():
        instances = pickle.load(instance_path.open('rb'))
    else:
        instances = reader.create_instance(logger, config.dir_dataset)
        pickle.dump(instances, instance_path.open('wb'))

    if args.do_train:
        if config.dataset in ['bach60']:
            seq_instances = instances
        else:
            seq_instances = reader.convert_to_chromaseq(instances, logger)
        train_instances, dev_instances, _ = reader.split_train_dev_test(seq_instances, logger)
        if config.model_to_initialize is not None:
            pretrained_checkpoint = torch.load(args.model_to_initialize, map_location=torch.device(args.device))
            model = NeuralHSMM(config)
            model.load_state_dict(pretrained_checkpoint['model'], strict=False)
            model.num_modes = config.num_modes  # overwrite num_modes
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
        if config.dataset in ['bach60']:
            seq_instances = instances
        else:
            seq_instances = reader.convert_to_chromaseq(instances, logger)
        _, _, test_instances = reader.split_train_dev_test(seq_instances, logger)
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
            accuracy_scores = run_evaluate_bach60(config, model, instance_for_test, logger)
        else:
            assert config.dataset in ['bach-4part-mxl']
            accuracy_scores = run_evaluate(config, model, model_filename.stem, instance_for_test, logger, output_dir)


if __name__ == '__main__':
    _args = add_arguments().parse_args()
    run(_args)


