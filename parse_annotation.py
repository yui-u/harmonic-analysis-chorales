import click
import json
import re
from pathlib import Path
from music21.interval import Interval
from music21.pitch import Pitch

from core.common.constants import SHARP_CIRCLE, FLAT_CIRCLE


@click.group()
def cli():
    pass


BACH_CHORALE_KEY_SIGS = {
    1: ['F#'],
    2: ['F#', 'C#', 'G#'],
    3: ['F#'],
    4: ['F#', 'C#', 'G#', 'D#'],
    5: ['F#'],
    6: ['B-'],
    7: ['F#', 'C#', 'G#'],
    8: ['B-', 'E-', 'A-'],
    9: ['F#'],
    10: [],
    11: [],
    12: [],
    13: [],
    14: ['F#'],
    15: [],
    16: ['F#', 'C#'],
    17: ['F#'],
    18: ['F#'],
    19: ['B-'],
    20: ['F#', 'C#']
}


@cli.command()
@click.option('-i', '--input_dir')
@click.option('-o', '--output_dir')
def parse_chorale_analyses(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir()
    ann_files = sorted(input_dir.glob('riemenschneider*.rntxt'))
    annotations = {}
    beats = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75]
    bar_pattern = r'b(\d+)'
    for ann_file in ann_files:
        riemen = int(ann_file.stem[-3:])
        assert riemen not in annotations
        key_sigs = BACH_CHORALE_KEY_SIGS[riemen]
        ann = {'key_sigs': key_sigs, 'comments': [], 'section': [],
               'annotation': {}, 'annotation_normalized': {}}

        if key_sigs:
            n_sigs = len(key_sigs)
            if key_sigs[0][-1] == '#':
                key_pitch = Pitch(SHARP_CIRCLE[n_sigs])
            else:
                key_pitch = Pitch(FLAT_CIRCLE[n_sigs])
        else:
            key_pitch = Pitch('C')
        interval = Interval(key_pitch, Pitch('C'))

        current_key, current_key_transposed = None, None
        with ann_file.open('r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if not line:
                pass
            elif line.startswith('Composer: '):
                composer = line[len('Composer: '):]
                assert composer == 'J. S. Bach'
                ann['composer'] = composer
            elif line.startswith('BWV: '):
                ann['bwv'] = line[len('BWV: '):]
            elif line.startswith('Title: '):
                ann['title'] = line[len('Title: '):]
            elif line.startswith('Analyst: '):
                ann['analyst'] = line[len('Analyst: '):]
            elif line.startswith('Proofreader: '):
                ann['proofreader'] = line[len('Proofreader: '):]
            elif line.startswith('Time Signature: '):
                ann['time-signature'] = line[len('Time Signature: '):]
                assert ann['time-signature'] in ['3/4', '4/4']
            elif line.startswith('Form: '):
                ann['form'] = line[len('Form: '):]
            elif line.startswith('Note: '):
                if 'email' not in line:
                    ann['comments'].append(line)
            elif line.startswith('m'):
                if 'var' in line:
                    ann['comments'].append(line)
                else:
                    ann_line = line.split(' ')
                    current_measure = int(ann_line[0][1:])
                    assert current_measure not in ann['annotation']
                    ann['annotation'][current_measure] = {}
                    ann['annotation_normalized'][current_measure] = {}
                    if re.match(bar_pattern, ann_line[1]):
                        bar = float(ann_line[1][1:])
                        start_item = 2
                    else:
                        bar = float(1)
                        start_item = 1
                    assert bar in beats and bar not in ann['annotation'][current_measure]
                    ann['annotation'][current_measure][bar] = []
                    ann['annotation_normalized'][current_measure][bar] = []

                    for m_ann in ann_line[start_item:]:
                        if m_ann.endswith(':'):
                            current_key = m_ann[:-1]
                            if 1 < len(current_key) and current_key[-1] == 'b':
                                current_key = current_key[:-1] + '-'
                            current_key_transposed = interval.transposePitch(Pitch(current_key)).name
                            if current_key[0].islower():
                                current_key_transposed = current_key_transposed.lower()
                        elif '||' in m_ann:
                            ann['section'].append((current_measure, bar))
                        elif re.match(bar_pattern, m_ann):
                            assert 1 < len(m_ann)
                            bar = float(m_ann[1:])
                            assert bar in beats and bar not in ann['annotation'][current_measure]
                            ann['annotation'][current_measure][bar] = []
                            ann['annotation_normalized'][current_measure][bar] = []
                        else:
                            if bool(m_ann):
                                ann['annotation'][current_measure][bar].append((current_key, m_ann))
                                ann['annotation_normalized'][current_measure][bar].append((current_key_transposed, m_ann))
        annotations[riemen] = ann
    with open(output_dir / Path('bach_chorale_annotation_m21.json'), 'w') as w:
        json.dump(annotations, w, indent=2)


if __name__ == '__main__':
    cli()
