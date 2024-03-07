import math

from typing import Dict

import numpy as np
from music21.corpus import parse
from music21.corpus.chorales import ChoraleListRKBWV
from music21.interval import Interval
from music21.pitch import Pitch
from music21.chord import Chord
from music21.note import Note, Rest
from music21.stream import Measure, Part
import torch

from core.common.constants import *
from core.preprocess.instances import ListInstance, ValueInstance

BACH_CHORALE_TRAIN_DEV_TEST_DICT_v3 = {
    1: TEST,
    2: TEST,
    3: TEST,
    4: TEST,
    5: TEST,
    6: TEST,
    7: TEST,
    8: TEST,
    9: TEST,
    10: TEST,
    11: TEST,
    12: TEST,
    13: TEST,
    14: TEST,
    15: TEST,
    16: TEST,
    17: TEST,
    18: TEST,
    19: TEST,
    20: TEST,
    21: DEV,
    22: TRAIN,
    23: TRAIN,
    24: TRAIN,
    25: TRAIN,
    26: TRAIN,
    27: TRAIN,
    28: TRAIN,
    29: DEV,
    30: TRAIN,
    31: TRAIN,
    32: TRAIN,
    33: TRAIN,
    34: TRAIN,
    35: TRAIN,
    36: TRAIN,
    37: TRAIN,
    38: TRAIN,
    39: TRAIN,
    40: TRAIN,
    41: TRAIN,
    42: TRAIN,
    43: TRAIN,
    44: TRAIN,
    45: DEV,
    46: DEV,
    47: TRAIN,
    48: TRAIN,
    49: TRAIN,
    50: TRAIN,
    51: DEV,
    52: DEV,
    53: DEV,
    54: TRAIN,
    55: TRAIN,
    56: TRAIN,
    57: TRAIN,
    58: TRAIN,
    59: TRAIN,
    60: TRAIN,
    61: TRAIN,
    62: DEV,
    63: TRAIN,
    64: TRAIN,
    65: DEV,
    66: TRAIN,
    67: TRAIN,
    68: DEV,
    69: TRAIN,
    70: TRAIN,
    71: TRAIN,
    72: TRAIN,
    73: TRAIN,
    74: TRAIN,
    75: DEV,
    76: TRAIN,
    77: DEV,
    78: DEV,
    79: TRAIN,
    80: TRAIN,
    81: TRAIN,
    82: TRAIN,
    83: TRAIN,
    84: TRAIN,
    85: TRAIN,
    86: TRAIN,
    87: TRAIN,
    88: TRAIN,
    89: DEV,
    90: TRAIN,
    91: TRAIN,
    92: DEV,
    93: TRAIN,
    94: DEV,
    95: TRAIN,
    96: TRAIN,
    97: TRAIN,
    98: TRAIN,
    99: TRAIN,
    100: TRAIN,
    101: TRAIN,
    102: TRAIN,
    103: TRAIN,
    104: TRAIN,
    105: TRAIN,
    106: DEV,
    107: TRAIN,
    108: TRAIN,
    109: TRAIN,
    110: TRAIN,
    111: DEV,
    112: TRAIN,
    113: TRAIN,
    114: DEV,
    115: TRAIN,
    116: TRAIN,
    117: TRAIN,
    118: TRAIN,
    119: TRAIN,
    120: TRAIN,
    121: DEV,
    122: TRAIN,
    123: TRAIN,
    124: TRAIN,
    125: TRAIN,
    126: TRAIN,
    127: TRAIN,
    128: TRAIN,
    129: TRAIN,
    130: TRAIN,
    131: TRAIN,
    132: TRAIN,
    133: TRAIN,
    134: TRAIN,
    135: TRAIN,
    136: TRAIN,
    137: TRAIN,
    138: TRAIN,
    139: TRAIN,
    140: DEV,
    141: TRAIN,
    142: TRAIN,
    143: DEV,
    144: TRAIN,
    145: DEV,
    146: TRAIN,
    147: TRAIN,
    148: TRAIN,
    149: TRAIN,
    150: TRAIN,
    151: TRAIN,
    152: TRAIN,
    153: TRAIN,
    154: TRAIN,
    155: DEV,
    156: DEV,
    157: TRAIN,
    158: DEV,
    159: TRAIN,
    160: TRAIN,
    161: DEV,
    162: TRAIN,
    163: TRAIN,
    164: DEV,
    165: TRAIN,
    166: TRAIN,
    167: TRAIN,
    168: TRAIN,
    169: TRAIN,
    170: TRAIN,
    171: TRAIN,
    172: TRAIN,
    173: DEV,
    174: TRAIN,
    175: TRAIN,
    176: DEV,
    177: TRAIN,
    178: TRAIN,
    179: DEV,
    180: DEV,
    181: TRAIN,
    182: DEV,
    183: TRAIN,
    184: TRAIN,
    185: DEV,
    186: TRAIN,
    187: TRAIN,
    188: TRAIN,
    189: TRAIN,
    190: TRAIN,
    191: TRAIN,
    192: DEV,
    193: DEV,
    194: TRAIN,
    195: TRAIN,
    196: TRAIN,
    197: TRAIN,
    198: TRAIN,
    199: DEV,
    200: TRAIN,
    201: TRAIN,
    202: TRAIN,
    203: TRAIN,
    204: TRAIN,
    205: TRAIN,
    206: TRAIN,
    207: TRAIN,
    208: TRAIN,
    209: TRAIN,
    210: TRAIN,
    211: DEV,
    212: TRAIN,
    213: TRAIN,
    214: DEV,
    215: TRAIN,
    216: DEV,
    217: DEV,
    218: DEV,
    219: TRAIN,
    220: DEV,
    221: TRAIN,
    222: TRAIN,
    223: DEV,
    224: TRAIN,
    225: DEV,
    226: TRAIN,
    227: DEV,
    228: TRAIN,
    229: TRAIN,
    230: TRAIN,
    231: DEV,
    232: TRAIN,
    233: TRAIN,
    234: DEV,
    235: TRAIN,
    236: TRAIN,
    237: TRAIN,
    238: TRAIN,
    239: TRAIN,
    240: DEV,
    241: TRAIN,
    242: TRAIN,
    243: DEV,
    244: TRAIN,
    245: TRAIN,
    246: TRAIN,
    247: TRAIN,
    248: TRAIN,
    249: TRAIN,
    250: TRAIN,
    251: DEV,
    252: TRAIN,
    253: DEV,
    254: TRAIN,
    255: TRAIN,
    256: TRAIN,
    257: TRAIN,
    258: DEV,
    259: TRAIN,
    260: TRAIN,
    261: TRAIN,
    262: TRAIN,
    263: TRAIN,
    264: DEV,
    265: TRAIN,
    266: TRAIN,
    267: TRAIN,
    268: DEV,
    269: TRAIN,
    270: TRAIN,
    271: TRAIN,
    272: TRAIN,
    273: TRAIN,
    274: TRAIN,
    275: TRAIN,
    276: TRAIN,
    277: TRAIN,
    278: TRAIN,
    279: TRAIN,
    280: TRAIN,
    281: TRAIN,
    282: TRAIN,
    283: TRAIN,
    284: TRAIN,
    285: TRAIN,
    286: DEV,
    287: TRAIN,
    288: TRAIN,
    289: TRAIN,
    290: TRAIN,
    291: DEV,
    292: DEV,
    293: TRAIN,
    294: TRAIN,
    295: TRAIN,
    296: TRAIN,
    297: TRAIN,
    298: DEV,
    299: TRAIN,
    300: DEV,
    301: TRAIN,
    302: DEV,
    303: TRAIN,
    304: TRAIN,
    305: TRAIN,
    306: TRAIN,
    307: TRAIN,
    308: TRAIN,
    309: TRAIN,
    310: TRAIN,
    311: TRAIN,
    312: TRAIN,
    313: DEV,
    314: TRAIN,
    315: DEV,
    316: TRAIN,
    317: TRAIN,
    318: TRAIN,
    319: TRAIN,
    320: DEV,
    321: TRAIN,
    322: TRAIN,
    323: DEV,
    324: TRAIN,
    325: DEV,
    326: TRAIN,
    327: TRAIN,
    328: TRAIN,
    329: DEV,
    330: TRAIN,
    331: TRAIN,
    332: TRAIN,
    333: TRAIN,
    334: DEV,
    335: TRAIN,
    336: TRAIN,
    337: TRAIN,
    338: TRAIN,
    339: DEV,
    340: DEV,
    341: TRAIN,
    342: TRAIN,
    343: TRAIN,
    344: TRAIN,
    345: TRAIN,
    346: TRAIN,
    347: DEV,
    348: TRAIN,
    349: TRAIN,
    350: TRAIN,
    351: TRAIN,
    352: TRAIN,
    353: TRAIN,
    354: DEV,
    355: TRAIN,
    356: TRAIN,
    357: TRAIN,
    358: TRAIN,
    359: TRAIN,
    360: TRAIN,
    361: DEV,
    362: TRAIN,
    363: TRAIN,
    364: TRAIN,
    365: TRAIN,
    366: TRAIN,
    367: TRAIN,
    368: TRAIN,
    369: TRAIN,
    370: TRAIN,
    371: TRAIN,
}

BACH_CHORALE_DUPLICATE_LIST = [
    [5, 309],
    [9, 361],
    [23, 88],
    [53, 178],
    [64, 256],
    [86, 195, 305],
    [91, 259],
    [93, 257],
    [100, 126],
    [120, 349],
    [125, 326],
    [131, 328],
    [144, 318],
    [156, 308],
    [198, 307],
    [199, 302],
    [201, 306],
    [235, 319],
    [236, 295],
    [248, 354],
    [254, 282],
    [313, 353],
]

BACH_CHORALE_INVALID_MEASURE_LIST = [
    130,
]


class BachChoralReaderBase(object):
    def __init__(self, config):
        self._config = config
        self._key_preprocessing = config.key_preprocessing
        self._train_dev_test_dict = BACH_CHORALE_TRAIN_DEV_TEST_DICT_v3

    @staticmethod
    def _get_ignore_duplicate_list():
        ignore_list = []
        for dpl in BACH_CHORALE_DUPLICATE_LIST:
            for d in dpl[1:]:  # choose smaller number
                ignore_list.append(d)
        ignore_list.extend(BACH_CHORALE_INVALID_MEASURE_LIST)
        return ignore_list

    def _get_scores(self, logger) -> Dict:
        bcl = ChoraleListRKBWV()
        ignore_list = self._get_ignore_duplicate_list()

        scores = {}
        cnt_allowed_timesig = 0
        cnt_not_duplicated = 0
        for k, v in bcl.byRiemenschneider.items():  # Riemenshneider contains 371 pieces
            riemenschneider = int(v['riemenschneider'])
            bwv_m21 = v['bwv']
            score = parse('bwv{}'.format(bwv_m21))
            parts = score.getElementsByClass(Part)
            timesig = score.parts[0].measure(2).getContextByClass('TimeSignature')
            cnt_allowed_timesig += 1

            if riemenschneider not in ignore_list:
                cnt_not_duplicated += 1

            num_measures = None
            added = False
            if len(parts) == 4:
                if (parts[0].id == 'Soprano' and
                        parts[1].id == 'Alto' and
                        parts[2].id == 'Tenor' and
                        parts[3].id == 'Bass'):  # four-part chorales only
                    # check consistency of key signatures of parts
                    key_sigs = []
                    for part in parts:
                        key_sigs.append(tuple(part.getElementsByClass(Measure)[0].keySignature.alteredPitches))
                    key_sigs = list(set(key_sigs))
                    if len(key_sigs) == 1 and riemenschneider not in ignore_list:
                        added = True
                        scores[riemenschneider] = {
                            'riemenschneider': riemenschneider,
                            'bwv': bwv_m21,
                            'score': score,
                            'key_sigs': list(key_sigs[0])}
                    # check the num measures
                    num_measures = len(parts[0].getElementsByClass(Measure))
            logger.info('|riemenschneider{}|bwv{}|{}|added={}'.format(
                riemenschneider, bwv_m21, num_measures, added))
        logger.info('Accepted time signature={}/371'.format(cnt_allowed_timesig))
        logger.info('Not duplicated={}/371'.format(cnt_not_duplicated))
        logger.info('Passed riemenschneiders={}/371'.format(len(scores)))
        return scores

    @staticmethod
    def _check_measure_alignment(transposed_score):
        assert (
            len(transposed_score.parts[0].getElementsByClass(Measure)) ==
            len(transposed_score.parts[1].getElementsByClass(Measure)) ==
            len(transposed_score.parts[2].getElementsByClass(Measure)) ==
            len(transposed_score.parts[3].getElementsByClass(Measure))
        )

    def _get_transposed_scores(self, score_item: Dict):
        score = score_item['score']
        key_sigs = score_item['key_sigs']
        if key_sigs:
            n_sigs = len(key_sigs)
            if key_sigs[0].name[-1] == '#':
                key_pitch = Pitch(SHARP_CIRCLE[n_sigs])
            else:
                key_pitch = Pitch(FLAT_CIRCLE[n_sigs])
        else:
            key_pitch = Pitch('C')

        transposed_scores = []
        shifts = []
        if self._key_preprocessing == KEY_PREPROCESS_NONE:
            interval = Interval(key_pitch, key_pitch)
            self._check_measure_alignment(score)
            transposed_scores.append(score)
            shifts.append(interval.pitchEnd.pitchClass - interval.pitchStart.pitchClass)
        elif self._key_preprocessing == KEY_PREPROCESS_NORMALIZE:
            interval = Interval(key_pitch, Pitch('C'))
            transposed_score = score.transpose(interval)
            self._check_measure_alignment(transposed_score)
            transposed_scores.append(transposed_score)
            shifts.append(interval.pitchEnd.pitchClass - interval.pitchStart.pitchClass)
        elif self._key_preprocessing == KEY_PREPROCESS_AUGMENTED:
            for pc in range(12):
                interval = Interval(key_pitch, Pitch(pitchClass=pc))
                transposed_score = score.transpose(interval)
                self._check_measure_alignment(transposed_score)
                transposed_scores.append(transposed_score)
                shifts.append(interval.pitchEnd.pitchClass - interval.pitchStart.pitchClass)
        else:
            raise NotImplementedError

        return transposed_scores, shifts


class BachChoraleReader(BachChoralReaderBase):
    def __init__(self, config):
        super().__init__(config)

    def _get_fermatas(self, score, chordify=True):
        if chordify:
            chords = score.flatten().chordify().notes
            fermatas = []
            for i, chord in enumerate(chords):
                for e in chord.expressions:
                    if e.name == 'fermata':
                        assert i not in fermatas
                        fermatas.append(i)
            return fermatas
        else:
            raise NotImplementedError

    @staticmethod
    def _align_offset(part_notes_raw, part_rests_raw, chord_offsets):
        part_notes = [part_notes_raw[0]]
        for pn in part_notes_raw[1:]:
            if pn.offset == part_notes[-1].offset:
                # if the number of notes in the part > 1, only one note is allowed
                pass
            else:
                part_notes.append(pn)

        if part_rests_raw:
            part_rests = {}
            for pr in part_rests_raw:
                assert pr.offset not in part_rests
                part_rests[pr.offset] = pr.duration
        else:
            part_rests = {}

        def isRest(_offset):
            is_rest = False
            for pr_offset, pr_duration in part_rests.items():
                if pr_offset <= _offset <= pr_offset + pr_duration.quarterLength:
                    is_rest = True
                    break
            return is_rest

        aligned_part_offset = {}
        cnt = 0
        for offset, chord in chord_offsets.items():
            if len(part_notes) <= cnt:
                if isRest(offset):
                    aligned_part_offset[offset] = Rest(offset=offset, duration=chord.duration)
                else:
                    aligned_part_offset[offset] = Note(part_notes[cnt - 1].pitch, offset=offset, duration=chord.duration)
            else:
                if offset == part_notes[cnt].offset:
                    assert offset not in aligned_part_offset
                    aligned_part_offset[offset] = Note(part_notes[cnt].pitch, offset=offset, duration=chord.duration)
                    cnt += 1
                else:
                    assert offset < part_notes[cnt].offset
                    if isRest(offset):
                        aligned_part_offset[offset] = Rest(offset=offset, duration=chord.duration)
                    else:
                        aligned_part_offset[offset] = Note(part_notes[cnt - 1].pitch, offset=offset, duration=chord.duration)
        assert len(aligned_part_offset) == len(chord_offsets)
        return aligned_part_offset

    def _get_aligned_satbs(self, score, fermatas):
        chords = score.flatten().chordify().getElementsByClass(Chord)

        fermata_metrics = [c.offset + c.duration.quarterLength for i, c in enumerate(chords) if i in fermatas]
        chord_offsets = dict([(c.offset, c) for c in chords])
        soprano_notes = score.parts[0].flatten().getElementsByClass(Note)
        soprano_rests = score.parts[0].flatten().getElementsByClass(Rest)
        alto_notes = score.parts[1].flatten().getElementsByClass(Note)
        alto_rests = score.parts[1].flatten().getElementsByClass(Rest)
        tenor_notes = score.parts[2].flatten().getElementsByClass(Note)
        tenor_rests = score.parts[2].flatten().getElementsByClass(Rest)
        bass_notes = score.parts[3].flatten().getElementsByClass(Note)
        bass_rests = score.parts[3].flatten().getElementsByClass(Rest)

        aligned_soprano = self._align_offset(soprano_notes, soprano_rests, chord_offsets)
        aligned_alto = self._align_offset(alto_notes, alto_rests, chord_offsets)
        aligned_tenor = self._align_offset(tenor_notes, tenor_rests, chord_offsets)
        aligned_bass = self._align_offset(bass_notes, bass_rests, chord_offsets)

        # append entire rests
        soprano_rests = dict([(pr.offset, pr.duration) for pr in soprano_rests])
        alto_rests = dict([(pr.offset, pr.duration) for pr in alto_rests])
        tenor_rests = dict([(pr.offset, pr.duration) for pr in tenor_rests])
        bass_rests = dict([(pr.offset, pr.duration) for pr in bass_rests])
        entire_rests_keys = set(soprano_rests.keys()) & set(alto_rests.keys()) & set(tenor_rests.keys()) & set(
            bass_rests.keys())
        entire_rests_keys = list(entire_rests_keys)
        for er in entire_rests_keys:
            assert soprano_rests[er] == alto_rests[er] == tenor_rests[er] == bass_rests[er]
            assert (er not in aligned_soprano) and (er not in aligned_alto) and (er not in aligned_tenor) and (
                        er not in aligned_bass)
            aligned_soprano[er] = Rest(offset=er, duration=soprano_rests[er])
            aligned_alto[er] = Rest(offset=er, duration=alto_rests[er])
            aligned_tenor[er] = Rest(offset=er, duration=tenor_rests[er])
            aligned_bass[er] = Rest(offset=er, duration=bass_rests[er])
            chord_offsets[er] = Rest(offset=er, duration=soprano_rests[er])

        return fermata_metrics, entire_rests_keys, chord_offsets, aligned_soprano, aligned_alto, aligned_tenor, aligned_bass

    def _get_chords_with_metrical_structure(self, score, fermatas):
        timesig = score.parts[0].measure(2).getContextByClass('TimeSignature')
        fermata_metrics, entire_rests_keys, chord_offsets, aligned_soprano, aligned_alto, aligned_tenor, aligned_bass = \
            self._get_aligned_satbs(score, fermatas)

        assert aligned_soprano.keys() == aligned_alto.keys() == aligned_tenor.keys() == aligned_bass.keys()
        aligned_keys = sorted(list(aligned_soprano.keys()))
        last_duration_max = max(
            [aligned_soprano[aligned_keys[-1]].duration.quarterLength,
             aligned_alto[aligned_keys[-1]].duration.quarterLength,
             aligned_tenor[aligned_keys[-1]].duration.quarterLength,
             aligned_bass[aligned_keys[-1]].duration.quarterLength]
        )
        metric_max = int(math.ceil(aligned_keys[-1] + last_duration_max))
        assert METRIC_BEAT_RATIO == 0.25
        if timesig.denominator == 2:
            metric_rate = METRIC_BEAT_RATIO * 2
        elif timesig.denominator == 4:
            metric_rate = METRIC_BEAT_RATIO
        else:
            raise NotImplementedError
        max_mc = int(1.0 / metric_rate)

        all_metrics = []
        all_chords = []
        all_beats = []
        all_pitch_histograms = []
        section_chords = []
        section_metrics = []
        section_beats = []
        section_pitch_histograms = np.zeros(12).astype(float)
        ak_count = 0
        for m in range(metric_max):
            for mc in range(max_mc):
                metric = m + metric_rate * mc
                if metric in fermata_metrics:
                    assert bool(section_chords)
                    assert bool(section_metrics)
                    assert bool(section_beats)
                    assert 0.0 < np.sum(section_pitch_histograms)
                    assert len(section_chords) == len(section_metrics) == len(section_beats)
                    all_chords.append(section_chords)
                    all_metrics.append(section_metrics)
                    all_beats.append(section_beats)
                    all_pitch_histograms.append(section_pitch_histograms)
                    section_chords = []
                    section_metrics = []
                    section_beats = []
                    section_pitch_histograms = np.zeros(12).astype(float)

                if ak_count < (len(aligned_keys) - 1) and aligned_keys[ak_count + 1] <= metric:
                    if ak_count < (len(aligned_keys) - 2) and aligned_keys[ak_count + 2] <= metric:
                        # There are few cases the chord has less duration than metric rate
                        print('Ignored chord the duration of which is less than metric_rate: ',
                              score.metadata.title,
                              aligned_keys[ak_count + 1],
                              chord_offsets[aligned_keys[ak_count + 1]])
                        if ak_count < (len(aligned_keys) - 3):
                            assert metric < aligned_keys[ak_count + 3]
                        ak_count += 2
                    else:
                        ak_count += 1

                k = aligned_keys[ak_count]
                satb = [
                    aligned_soprano[k],
                    aligned_alto[k],
                    aligned_tenor[k],
                    aligned_bass[k]
                ]
                # check the alignment
                if k not in entire_rests_keys:
                    satb_set = sorted(list(set([n.pitch.midi for n in satb if not n.isRest])))
                    chord_set = sorted(list(set([n.pitch.midi for n in chord_offsets[k].notes])))
                    if satb_set != chord_set:
                        # there are very few cases satb_set do not match chord_set
                        print("Ignore inconsistencies: ", satb_set, chord_set)
                section_chords.append(satb)
                section_metrics.append(metric)
                section_beats.append(score.beatAndMeasureFromOffset(metric)[0])
                for n in satb:
                    if not n.isRest:
                        section_pitch_histograms[n.pitch.pitchClass] += metric_rate

        if section_chords:
            assert bool(section_metrics)
            assert bool(section_beats)
            assert len(section_chords) == len(section_metrics) == len(section_beats)
            if np.sum(section_pitch_histograms) <= 0.0:
                for sc in section_chords:
                    for scp in sc:
                        assert scp.isRest
                # if all items are rest, don't append
            else:
                all_chords.append(section_chords)
                all_metrics.append(section_metrics)
                all_beats.append(section_beats)
                all_pitch_histograms.append(section_pitch_histograms)
            if metric + metric_rate != fermata_metrics[-1]:
                # there are very few cases missing last fermata
                print('Missing the last fermata: ', score.metadata.title)

        assert max([max(sb) for sb in all_beats]) < (timesig.numerator + 1)
        assert ak_count == len(aligned_keys) - 1
        assert len(all_chords) == len(all_metrics) == len(all_beats) == len(all_pitch_histograms)
        return all_chords, all_metrics, all_beats, all_pitch_histograms, metric_rate, timesig

    @staticmethod
    def _convert_to_binary_chord(chord):
        bin_chord = [0] * 12
        for p in chord:
            assert bin_chord[p] == 0
            bin_chord[p] = 1
        return bin_chord

    def _get_length_per_measure(self, timesig, min_duration):
        if timesig.numerator == 4 and timesig.denominator == 4:
            len_per_measure = int(4 / min_duration)
            assert len_per_measure % 4 == 0
            len_per_beat = int(len_per_measure / 4)
        elif timesig.numerator == 3 and timesig.denominator == 4:
            len_per_measure = int(3 / min_duration)
            assert len_per_measure % 3 == 0
            len_per_beat = int(len_per_measure / 3)
        elif timesig.numerator == 3 and timesig.denominator == 2:
            len_per_measure = int(6 / min_duration)
            assert len_per_measure % 3 == 0
            len_per_beat = int(len_per_measure / 3)
        elif timesig.numerator == 12 and timesig.denominator == 8:
            len_per_measure = int(6 / min_duration)
            assert len_per_measure % 4 == 0
            len_per_beat = int(len_per_measure / 4)
        else:
            raise NotImplementedError
        return len_per_measure, len_per_beat

    def create_instance(self, logger, vocab=None):
        if vocab is not None:
            pad_binary_pitch_class = vocab.pad_binary_pitch_class
        else:
            pad_binary_pitch_class = [0] * 12
        pad_midi = [0] * 128
        config = self._config
        logger.info('Create instances')
        scores = self._get_scores(logger)
        instances_raw = []
        for riemen, item in scores.items():
            score = item['score']
            bwv = item['bwv']
            fermatas = self._get_fermatas(score)
            transposed_scores, shifts = self._get_transposed_scores(item)
            for transposed_score, shift in zip(transposed_scores, shifts):
                satbs, metrics, beats, pitch_histograms, metric_rate, timesig = \
                    self._get_chords_with_metrical_structure(transposed_score, fermatas)
                if vocab is not None:
                    # vocab update
                    for section_satbs, section_metrics in zip(satbs, metrics):
                        for satb in section_satbs:
                            sorted_pitches = sorted(list(set([n.pitch.pitchClass for n in satb if not n.isRest])))
                            vocab.update_counts(sorted_pitches, metric_rate)
                len_per_measure, len_per_beat = self._get_length_per_measure(timesig, metric_rate)
                num_sections = len(satbs)
                # add instance
                for section_id, (section_satbs, section_metrics, section_beats, section_histogram) in enumerate(
                        zip(satbs, metrics, beats, pitch_histograms)):
                    section_chord_beats = [_ for _ in section_beats]
                    section_pitch_classes = \
                        [sorted(list(set([n.pitch.pitchClass for n in satb if not n.isRest]))) for satb in section_satbs]
                    section_binary_pitches = [self._convert_to_binary_chord(pitch_classes) for pitch_classes in section_pitch_classes]
                    section_midi_numbers = []
                    section_midis = []
                    section_bass_pcs = []
                    section_part_midis = [[] for _ in range(4)]
                    for satb in section_satbs:
                        midi_numbers = [n.pitch.midi if not n.isRest else REST_INDEX for n in satb]
                        midi_notes = [n.pitch.midi for n in satb if not n.isRest]
                        if midi_notes:
                            section_bass_pcs.append(min(midi_notes) % 12)
                        else:
                            section_bass_pcs.append(REST_INDEX)
                        section_midi_numbers.append(midi_numbers)
                        midi_vec = torch.zeros(128).long()
                        for part_id, part_midi in enumerate(midi_numbers):
                            part_midi_vec = torch.zeros(128).long()
                            if part_midi != REST_INDEX:
                                part_midi_vec[part_midi] += 1
                                midi_vec[part_midi] += 1
                            part_midi_vec = (part_midi_vec > 0).long()  # binarize
                            section_part_midis[part_id].append(part_midi_vec.tolist())
                        midi_vec = (midi_vec > 0).long()  # binarize
                        section_midis.append(midi_vec.tolist())
                    sequence_length = len(section_binary_pitches)
                    assert (sequence_length ==
                            len(section_binary_pitches) ==
                            len(section_midi_numbers) ==
                            len(section_part_midis[0]) ==
                            len(section_part_midis[1]) ==
                            len(section_part_midis[2]) ==
                            len(section_part_midis[3]) ==
                            len(section_midis) ==
                            len(section_bass_pcs) ==
                            len(section_chord_beats) ==
                            len(section_metrics)), (
                        len(section_binary_pitches),
                        len(section_midi_numbers),
                        len(section_part_midis[0]),
                        len(section_midis),
                        len(section_bass_pcs),
                        len(section_chord_beats),
                        len(section_metrics)
                    )
                    if sequence_length <= config.max_sequence_length:
                        section_binary_pitches += \
                            [pad_binary_pitch_class for _ in range(config.max_sequence_length - sequence_length)]
                        section_midi_numbers += \
                            [[PAD_INDEX] * 4 for _ in range(config.max_sequence_length - sequence_length)]
                        section_midis += [pad_midi for _ in range(config.max_sequence_length - sequence_length)]
                        section_bass_pcs += [PAD_INDEX for _ in range(config.max_sequence_length - sequence_length)]
                        section_chord_beats += \
                            [BEAT_SPECIAL_SYMBOL for _ in range(config.max_sequence_length - sequence_length)]
                        for part_id in range(4):
                            section_part_midis[part_id] += \
                                [pad_midi for _ in range(config.max_sequence_length - sequence_length)]
                    else:
                        section_binary_pitches = section_binary_pitches[:config.max_sequence_length]
                        section_midi_numbers = section_midi_numbers[:config.max_sequence_length]
                        section_midis = section_midis[:config.max_sequence_length]
                        section_bass_pcs = section_bass_pcs[:config.max_sequence_length]
                        section_chord_beats = section_chord_beats[:config.max_sequence_length]
                        for part_id in range(4):
                            section_part_midis[part_id] = section_part_midis[part_id][:config.max_sequence_length]
                        sequence_length = config.max_sequence_length

                    instance = {
                        'observation_chroma': ListInstance(list_instances=section_binary_pitches),
                        'observation_part_midi_binary': ListInstance(list_instances=section_part_midis),
                        'observation_part_midi_indices': ListInstance(list_instances=section_midi_numbers),
                        'beat': ListInstance(list_instances=section_chord_beats),
                        'sequence_length': ValueInstance(sequence_length),
                        'timesignature_numerator': ValueInstance(int(timesig.numerator)),
                        'timesignature_denominator': ValueInstance(int(timesig.denominator)),
                        META_DATA: {
                            'reader_name': self.__class__.__name__,
                            'bwv': bwv,
                            'riemenschneider': riemen,
                            'key_sigs': item['key_sigs'],
                            'shift': shift,
                            'bass': section_bass_pcs,
                            'section_id': section_id,
                            'num_sections': num_sections,
                            'length_per_measure': len_per_measure,
                            'length_per_beat': len_per_beat,
                            'min_duration': metric_rate,
                            'metrics': section_metrics
                        }
                    }
                    instances_raw.append(instance)
        return instances_raw

    def split_train_dev_test(self, instances, logger):
        train_instances, dev_instances, test_instances = [], [], []
        train_pieces = []
        dev_pieces = []
        test_pieces = []
        counted_riemens = []
        for instance in instances:
            riemen = instance[META_DATA]['riemenschneider']
            if self._train_dev_test_dict[riemen] == TRAIN:
                train_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    train_pieces.append(riemen)
            elif self._train_dev_test_dict[riemen] == DEV:
                dev_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    dev_pieces.append(riemen)
            else:
                assert self._train_dev_test_dict[riemen] == TEST
                test_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
                    test_pieces.append(riemen)
        logger.info("{}-pieces={}, {}-pieces={}, {}-pieces={}".format(
            TRAIN, len(train_pieces), DEV, len(dev_pieces), TEST, len(test_pieces)))
        logger.info("{}-phrases={}, {}-phrases={}, {}-phrases={}".format(
            TRAIN, len(train_instances), DEV, len(dev_instances), TEST, len(test_instances)))
        return train_instances, dev_instances, test_instances

