import bisect
import copy
import math
from pathlib import Path

import music21 as m21
from music21.interval import Interval
from music21.pitch import Pitch
from music21.chord import Chord
from music21.note import Note, Rest
from music21.stream import Measure, Voice
from music21.corpus.chorales import ChoraleListRKBWV

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from core.common.constants import *

from core.preprocess.instances import ListInstance, ValueInstance

KEY_DE2EN = {
    'c': 'c',
    'cis': 'c#',
    'des': 'd-',
    'd': 'd',
    'dis': 'd#',
    'es': 'e-',
    'e': 'e',
    'f': 'f',
    'fis': 'f#',
    'ges': 'g-',
    'g': 'g',
    'gis': 'g#',
    'as': 'a-',
    'a': 'a',
    'ais': 'a#',
    'b': 'b-',
    'h': 'b',
    'his': 'b#',
    'ces': 'c-',
}

DEGREES = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']

NOTE_FEAT = {
    'part': 0,
    'voice': 1,
    'offset': 2,
    'beat': 3,
    'length': 4,
    'midi': 5,
    'pc': 6
}

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
    130,  # It is in 4/4, but the eighth measure is only 2 x quarter long.
]


class MxlReader(object):
    def __init__(self, config):
        self._small_eps = 1e-13
        self._large_eps = 1e-5
        self._config = config
        self._key_preprocessing = config.key_preprocessing

    @staticmethod
    def _check_key_signatures(score):
        m_key_signatures = {}
        for pi in range(len(score.parts)):
            for mi, m in enumerate(score.parts[pi].getElementsByClass(Measure)):
                if hasattr(m, 'keySignature') and m.keySignature is not None:
                    if mi not in m_key_signatures:
                        m_key_signatures[mi] = []
                    m_key_signatures[mi].append(tuple(m.keySignature.alteredPitches))

        valid = True
        for k, v in m_key_signatures.items():
            if len(list(set(v))) != 1:
                valid = False
            m_key_signatures[k] = list(set(v))[0]
        return m_key_signatures, valid

    @staticmethod
    def _check_measure_alignment(score):
        ms = [(m.number, m.offset) for m in score.parts[0].getElementsByClass(Measure)]
        for pi in range(1, len(score.parts)):
            if ms != [(m.number, m.offset) for m in score.parts[pi].getElementsByClass(Measure)]:
                print('Inconsistent measures')
                return False
        return True

    def _get_transposed_score(self, score, key_sigs):
        if key_sigs:
            n_sigs = len(key_sigs)
            if key_sigs[0].name[-1] == '#':
                key_pitch = Pitch(SHARP_CIRCLE[n_sigs])
            else:
                key_pitch = Pitch(FLAT_CIRCLE[n_sigs])
        else:
            key_pitch = Pitch('C')

        interval = Interval(key_pitch, Pitch('C'))
        transposed_score = score.transpose(interval)

        return transposed_score, interval

    def _get_resolution(self, score):
        timesig = score.parts[0].measure(2).getContextByClass('TimeSignature')
        # resolution
        if self._config.resolution_str == '8th':
            resolution = 0.5
        elif self._config.resolution_str == '16th':
            resolution = 0.25
        elif self._config.resolution_str == 'half-beat':
            if '{}/{}'.format(timesig.numerator, timesig.denominator) in ['2/2', '3/2', '3/4', '4/4']:
                resolution = (4.0 / float(timesig.denominator)) * 0.5
            elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['6/8', '12/8']:
                resolution = 0.5
            else:
                raise NotImplementedError
        elif self._config.resolution_str == 'beat':
            if '{}/{}'.format(timesig.numerator, timesig.denominator) in ['2/2', '3/2', '3/4', '4/4']:
                resolution = 4.0 / float(timesig.denominator)
            elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['6/8', '12/8']:
                resolution = 1.5
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # time per measure
        if '{}/{}'.format(timesig.numerator, timesig.denominator) in ['2/2', '3/2', '3/4', '4/4']:
            time_per_measure = (4.0 * timesig.numerator) / float(timesig.denominator)
        elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['6/8', '12/8']:
            time_per_measure = 0.5 * timesig.numerator
        else:
            raise NotImplementedError

        return resolution, time_per_measure

    @staticmethod
    def _get_notes(score):
        notes = {}
        last_note_offset = 0.0
        for pi, part in enumerate(score.parts):
            for m in part.getElementsByClass(Measure):
                m_iter = [v for v in m.getElementsByClass(Voice)]
                if bool(m.elements):
                    m_iter += [m]
                for vi, v in enumerate(m_iter):
                    if (pi, vi) not in notes:
                        notes[(pi, vi)] = []
                    for n in [vn for vn in v.elements if isinstance(vn, Note) or isinstance(vn, Rest) or isinstance(vn, Chord)]:
                        if 0.0 < n.quarterLength:  # ignore invalid notes
                            notes[(pi, vi)].append((m, n))
                            if last_note_offset < m.offset + n.offset + n.quarterLength:
                                last_note_offset = m.offset + n.offset + n.quarterLength

        # sort key
        notes = dict(sorted(notes.items()))
        # sort value
        for pk, pv in notes.items():
            pvo = []
            for v in pv:
                n_offset = min([n.offset for n in v[1].notes]) if isinstance(v[1], Chord) else v[1].offset
                pvo.append((v[0].offset + n_offset, v))
            pvo = sorted(pvo, key=lambda x:x[0])
            notes[pk] = [v[1] for v in pvo]
        return notes, last_note_offset

    def _get_score_as_graph(self, score):
        resolution, time_per_measure = self._get_resolution(score)
        notes, last_note_offsets = self._get_notes(score)
        timesteps = [i for i in range(int(last_note_offsets / resolution) + 1)]

        data = HeteroData()

        # time nodes
        data['time'].node_id = torch.arange(len(timesteps))
        time_feature = torch.zeros((len(timesteps), 1))
        for i in range(len(timesteps)):
            time_feature[i, 0] = float(i * resolution)
        data['time'].x = time_feature

        # note nodes and h-notes edges (self and reverse included)
        nf = NOTE_FEAT.copy()
        note_feature = [[] for _ in range(len(nf))]
        raw_ann = {}
        note_id = 0
        offset2mn = {}
        fermata_positions = []
        for pk, part_mn in notes.items():
            for pmn in part_mn:
                cur_note_ids = []
                m, pn = pmn
                if isinstance(pn, Chord):
                    pns = pn.notes
                else:
                    pns = [pn]
                for n in pns:
                    # timestep to measure, note offsets
                    if m.offset + n.offset in offset2mn:
                        assert (
                            (offset2mn[m.offset + n.offset]['m_number'] == m.number) and
                            (offset2mn[m.offset + n.offset]['m_paddingLeft'] == m.paddingLeft) and
                            (offset2mn[m.offset + n.offset]['m_offset'] == m.offset) and
                            (offset2mn[m.offset + n.offset]['n_offset'] == n.offset)
                        )
                        if math.isnan(offset2mn[m.offset + n.offset]['beat']) and not math.isnan(n.beat):
                            offset2mn[m.offset + n.offset]['beat'] = n.beat
                    else:
                        offset2mn[m.offset + n.offset] = {
                            'm_number': m.number, 'm_paddingLeft': m.paddingLeft, 'm_offset': m.offset,
                            'n_offset': n.offset, 'beat': n.beat}
                    # fermata
                    for e in n.expressions:
                        if e.name == 'fermata':
                            if m.offset + n.offset not in fermata_positions:
                                fermata_positions.append(m.offset + n.offset + n.quarterLength)
                    # note features
                    note_feature[nf['part']].append(pk[0])
                    note_feature[nf['voice']].append(pk[1])
                    note_feature[nf['offset']].append(float(m.offset + n.offset))
                    note_feature[nf['beat']].append(float(n.beat) if not math.isnan(n.beat) else 0.0)
                    note_feature[nf['length']].append(float(n.quarterLength))
                    if isinstance(n, Rest):
                        note_feature[nf['midi']].append(REST_INDEX)
                        note_feature[nf['pc']].append(REST_INDEX)
                    else:
                        note_feature[nf['midi']].append(n.pitch.midi)
                        note_feature[nf['pc']].append(n.pitch.midi % 12)
                    # update note_id
                    cur_note_ids.append(note_id)
                    note_id += 1

        num_notes = len(note_feature[0])
        data['note'].node_id = torch.arange(num_notes)
        data['note'].x = torch.tensor(note_feature).transpose(0, 1)

        # time to note edges
        note2time_edges = [[], []]
        note2time_edge_feat = []
        for t in range(len(timesteps) - 1):
            t_start = t * resolution
            t_end = (t + 1) * resolution
            for note_id in range(num_notes):
                n_start = data['note'].x[note_id, 2].item()
                n_end = n_start + data['note'].x[note_id, 4].item()
                # range of overlap
                overlap = min(t_end, n_end) - max(t_start, n_start)
                if 0.0 < overlap:
                    note2time_edges[0].append(note_id)
                    note2time_edges[1].append(t)
                    note2time_edge_feat.append(overlap)

        data['note', 'n2t', 'time'].edge_index = torch.tensor(note2time_edges)
        data['note', 'n2t', 'time'].edge_attr = torch.tensor(note2time_edge_feat)

        raw_ann = dict(sorted(raw_ann.items(), key=lambda x: x[0]))
        return data, resolution, time_per_measure, offset2mn, raw_ann, fermata_positions

    def create_instance(self, logger, dir_dataset):
        logger.info('Create instance')
        if '4part' in self._config.dataset:
            duplicated_list = []
            for dpl in BACH_CHORALE_DUPLICATE_LIST:
                for d in dpl[1:]:  # choose smaller number
                    duplicated_list.append(d)
            mxl_files = ChoraleListRKBWV().byRiemenschneider
        else:
            dir_mxl = Path(dir_dataset)
            mxl_files = sorted(dir_mxl.glob("*.musicxml"))
        instances = []
        read_items = 0
        accepted_items = 0
        for mxl in mxl_files:
            read_items += 1
            if '4part' in self._config.dataset:
                v = mxl_files[mxl]
                org_score = m21.corpus.parse('bwv{}'.format(v['bwv']))
                mxl_filename = 'riemen{}-bwv{}'.format(v['riemenschneider'], v['bwv'])
                ignored = False
                if int(v['riemenschneider']) in duplicated_list:
                    ignored = True
                    logger.info('Duplicated 4part chorale: {}'.format(mxl_filename))
                if (len(org_score.parts) != 4) or not (
                        org_score.parts[0].id == 'Soprano' and
                        org_score.parts[1].id == 'Alto' and
                        org_score.parts[2].id == 'Tenor' and
                        org_score.parts[3].id == 'Bass'):
                    ignored = True
                    logger.info('Not ``4part" chorale: {}, {}'.format(len(org_score.parts), '-'.join(p.id for p in org_score.parts)))
                if int(v['riemenschneider']) in BACH_CHORALE_INVALID_MEASURE_LIST:
                    ignored = True
                    logger.info('Has invalid measures: {}'.format(mxl_filename))
            else:
                org_score = m21.converter.parse(mxl)
                mxl_filename = mxl.stem
                ignored = False

            # Allow the key signature to change during the piece.
            # However, it is not allowed to have different key signatures between parts.
            # Currently, transposing instruments are not supported.
            m_key_signatures, valid_key_sig = self._check_key_signatures(org_score)

            if ignored or (not self._check_measure_alignment(org_score)) or (not valid_key_sig):
                logger.info('Ignore {}'.format(mxl_filename))
                continue

            if self._key_preprocessing == KEY_PREPROCESS_NORMALIZE:
                # Not allow the key signature to change during the piece when key_preprocessing == NORMALIZE
                if 1 < len(set(m_key_signatures.values())):
                    continue

            accepted_items += 1
            logger.info('Reading {}'.format(mxl_filename))

            key_sigs = list(m_key_signatures.values())[0]
            key_sigs = [ks.name for ks in key_sigs]
            if self._key_preprocessing == KEY_PREPROCESS_NORMALIZE:
                score, interval = self._get_transposed_score(org_score, list(m_key_signatures.values())[0])
            else:
                score = org_score
                interval = Interval(0)
            graph, resolution, time_per_measure, offset2mn, raw_ann, fermata_positions = self._get_score_as_graph(score)
            fermata_positions = sorted(set(fermata_positions))
            instances.append({
                META_DATA: {
                    'filename': mxl_filename,
                    'key_sigs': ','.join(key_sigs),
                    'interval': str(interval.directedName),
                    'resolution': resolution,
                    'time_per_measure': time_per_measure,
                    'offset2mn': offset2mn,
                    'ann_raw': raw_ann,
                    'fermata': fermata_positions
                },
                'graph': graph
            })
        return instances

    def convert_to_chromaseq(self, instances, logger, segment_by_fermata=True, min_measure_segment=1.0):
        logger.info('Create chroma sequence.')
        chroma_dim = 12
        instances_seq = []
        for instance in instances:
            graph = instance['graph']
            time_resolution = instance[META_DATA]['resolution']
            num_edges = graph['note', 'n2t', 'time'].edge_index.size(-1)
            timesteps = graph['time'].x.squeeze(-1).tolist()[:-1]
            x_pc_index = graph['note'].x[:, NOTE_FEAT['pc']].long()
            x_pc_index = torch.where(
                x_pc_index < 0,
                torch.ones_like(x_pc_index) * chroma_dim,  # dummy rest
                x_pc_index
            )
            x_pc = F.one_hot(x_pc_index, num_classes=chroma_dim + 1)[:, :-1]  # (N, C+1) -> (N, C)
            x_chroma = torch.zeros((len(timesteps), chroma_dim))  # (L, C)
            x_bass = torch.ones(len(timesteps)).long() * PAD_INDEX  # (L,)
            for e in range(num_edges):
                e_note = graph['note', 'n2t', 'time'].edge_index[0, e].item()
                e_time = graph['note', 'n2t', 'time'].edge_index[1, e].item()
                note_len = min(graph['note'].x[e_note, NOTE_FEAT['length']].item(), time_resolution)
                note_len /= time_resolution
                x_chroma[e_time] = x_chroma[e_time] + (x_pc[e_note] * note_len)
                if x_bass[e_time] == PAD_INDEX or graph['note'].x[e_note, NOTE_FEAT['midi']].item() < x_bass[e_time]:
                    x_bass[e_time] = graph['note'].x[e_note, NOTE_FEAT['midi']].item()
            if segment_by_fermata:
                if bool(instance[META_DATA]['fermata']):
                    fermata_positions = []
                    prev_fermata_position = 0.0
                    for fmp in sorted(instance[META_DATA]['fermata']):
                        # avoid too short segment
                        if min_measure_segment * instance[META_DATA]['time_per_measure'] <= fmp - prev_fermata_position:
                            fermata_positions.append(fmp)
                            prev_fermata_position = fmp
                    if (timesteps[-1] - fermata_positions[-1]) < min_measure_segment * instance[META_DATA]['time_per_measure']:
                        fermata_positions = fermata_positions[:-1]  # Prevent the last segment from being too short
                    fermata_position_indices = [0]
                    for fmp in fermata_positions:
                        if fmp in timesteps:
                            fermata_position_indices.append(timesteps.index(fmp))
                        else:
                            fermata_position_indices.append(bisect.bisect_right(timesteps, fmp))
                    fermata_position_indices.append(len(timesteps))
                else:
                    fermata_position_indices = [0, len(timesteps)]
                for si, (fs, fe) in enumerate(zip(fermata_position_indices[:-1], fermata_position_indices[1:])):
                    seg_timesteps = timesteps[fs:fe]
                    seg_x_chroma = x_chroma[fs:fe]
                    seg_x_bass = x_bass[fs:fe]
                    if self._config.max_sequence_length < len(seg_timesteps):
                        seg_timesteps = seg_timesteps[:self._config.max_sequence_length]
                        seg_x_chroma = seg_x_chroma[:self._config.max_sequence_length]
                        seg_x_bass = seg_x_bass[:self._config.max_sequence_length]
                    seg_metadata = copy.deepcopy(instance[META_DATA])
                    seg_metadata['timestep'] = seg_timesteps
                    seg_metadata['seg_index'] = si
                    seg_metadata['num_segments'] = len(fermata_position_indices) - 1
                    instance_seq = {
                        META_DATA: seg_metadata,
                        'sequence_length': ValueInstance(len(seg_timesteps)),
                        'x_chroma': ListInstance(seg_x_chroma.tolist(), pad_value=0),
                        'x_bass': ListInstance(seg_x_bass.tolist(), pad_value=PAD_INDEX)
                    }
                    instances_seq.append(instance_seq)
            else:
                if self._config.max_sequence_length < len(timesteps):
                    timesteps = timesteps[:self._config.max_sequence_length]
                    x_chroma = x_chroma[:self._config.max_sequence_length]
                    x_bass = x_bass[:self._config.max_sequence_length]
                instance[META_DATA]['timestep'] = timesteps
                instance_seq = {
                    META_DATA: instance[META_DATA],
                    'sequence_length': ValueInstance(len(timesteps)),
                    'x_chroma': ListInstance(x_chroma.tolist(), pad_value=0),
                    'x_bass': ListInstance(x_bass.tolist(), pad_value=PAD_INDEX)
                }
                instances_seq.append(instance_seq)
        return instances_seq

    def _split_train_dev_test_cv(self, instances, logger):
        mxl_filenames = sorted(list(set(instance[META_DATA]['filename'] for instance in instances)))
        split_no_dict = dict([(fn, i % self._config.cv_num_set) for i, fn in enumerate(mxl_filenames)])
        train_instances = []
        dev_instances = []
        test_instances = []
        for instance in instances:
            fn = instance[META_DATA]['filename']
            split_no = split_no_dict[fn]
            instance[META_DATA]['cv_set_no'] = split_no
            if split_no == self._config.cv_set_no:
                test_instances.append(instance)
            elif (split_no - 1) % self._config.cv_num_set == self._config.cv_set_no:
                dev_instances.append(instance)
            else:
                train_instances.append(instance)
        logger.info("{}-sequences={}, {}-sequences={}, {}-sequences={}".format(
            TRAIN, len(train_instances), DEV, len(dev_instances), TEST, len(test_instances)))
        return train_instances, dev_instances, test_instances

    def _split_train_dev_test_4part(self, instances, logger):
        train_instances, dev_instances, test_instances = [], [], []
        counted_riemens = []
        for instance in instances:
            riemen = int(instance[META_DATA]['filename'].split('-')[0][len('riemen'):])
            if BACH_CHORALE_TRAIN_DEV_TEST_DICT_v3[riemen] == TRAIN:
                train_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
            elif BACH_CHORALE_TRAIN_DEV_TEST_DICT_v3[riemen] == DEV:
                dev_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
            else:
                assert BACH_CHORALE_TRAIN_DEV_TEST_DICT_v3[riemen] == TEST
                test_instances.append(instance)
                if riemen not in counted_riemens:
                    counted_riemens.append(riemen)
        logger.info("{}-pieces={}, {}-pieces={}, {}-pieces={}".format(
            TRAIN, len(train_instances), DEV, len(dev_instances), TEST, len(test_instances)))
        return train_instances, dev_instances, test_instances

    def split_train_dev_test(self, instances, logger):
        if '4part' in self._config.dataset:
            train_instances, dev_instances, test_instances = self._split_train_dev_test_4part(instances, logger)
        else:
            train_instances, dev_instances, test_instances = self._split_train_dev_test_cv(instances, logger)
        return train_instances, dev_instances, test_instances
