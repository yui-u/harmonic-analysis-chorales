# Unsupervised Harmonic Analysis with Code Quality Templates
This repository is the implementation of 
["Unsupervised Learning of Harmonic Analysis Based on Neural HSMM with Code Quality Templates"](http://arxiv.org/abs/2403.04135)
(Uehara; Proceedings of the 11th International Conference on New Music Concepts, pp.30--53).

## Requirements
- Python 3.10.10
- Required packages are listed in: [requirements.txt](requirements.txt)

## Dataset
### (bach60) Bach Chorales dataset formatted by Radicioni and Esposito [1]
Download the dataset from: https://archive.ics.uci.edu/dataset/298/bach+choral+harmony
, and place `jsbach_chorals_harmony.names` and `jsbach_chorals_harmony.data` in the `harmonic-analysis-chorales/bach60`.

[1] D. P. Radicioni and R. Esposito. BREVE: An HMPerceptron-based Chord Recognition System. 
In Advances in Music Information Retrieval, pages 143–164. 
Springer Berlin Heidelberg, 2010.

We followed the original 10-fold cross-validation split provided by Masada and Bunescu [2].
Information on this splitting was obtained here: https://github.com/kristenmasada/chord_recognition_semi_crf/tree/master/folds

[2] K. Masada and R. Bunescu. Chord Recognition in Symbolic Music:
A Segmental CRF Model, Segment-Level Features, 
and Comparative Evaluations on Classical and Popular music. 
Transactions of the International Society for Music Information Retrieval, 2(1):1–13, 2019.

Download the `folds` and place it in the `harmonic-analysis-chorales/bach60`.

### (bach-4part-mxl) 371 chorales in MusicXML format from the Music21 Library
This dataset is included in the Music21 Library and can be used by installing music21.
A human analysis for evaluation is available here: https://github.com/cuthbertLab/music21/tree/master/music21/corpus/bach/choraleAnalyses

Pre-process the analysis with the following command
and place the resulting file (`bach_chorale_annotation_m21.json`) in `harmonic-analysis-chorales/human_annotation_music21`.
```
python parse_annotation.py parse-chorale-analyses -i <downloaded-original-data> -o human_annotation_music21
```

## Training
### Training Phase1
(Pre-)Training the model with the normalized data and the no_shift setting.
If GPU is not available, change to `--device cpu`.
#### bach60
- To perform cross-validation, set cv_set_no to 0-9 and run each.
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123  \
--num_modes 2 \
--inter_key_transition_limit 0.0 \
--no_shift \
--num_epochs 480 \
--key_preprocessing key-preprocess-normalized \
--dataset bach60 \
--batch_size 2 \
--cv_set_no 0
```

#### bach-4part-mxl
- `bach-4part-mxl` uses a fixed train, validation, and test split.
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123 \
--num_modes 2 \
--inter_key_transition_limit 0.0 \
--no_shift \
--num_epochs 480 \
--key_preprocessing key-preprocess-normalized \
--dataset bach-4part-mxl \
--batch_size 8
```

### Training Phase2
Additional training with original data without normalization.
Shift and inter_key_transition are enabled.
In the Phase 2 training, the model learned in Phase 1 is used as the initial value.

#### bach60
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123 \
--num_modes 2 \
--inter_key_transition_limit 0.01 \
--num_epochs 240 \
--key_preprocessing key-preprocess-none \
--dataset bach60 \
--batch_size 2 \
--cv_set_no 0 \
--model_to_initialize <the-model-trained-in-phase1>
```

#### bach-4part-mxl
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123 \
--num_modes 2 \
--inter_key_transition_limit 0.01 \
--num_epochs 240 \
--key_preprocessing key-preprocess-none \
--dataset bach-4part-mxl \
--batch_size 8 \
--model_to_initialize <the-model-trained-in-phase1>
```

### Trained models
[bach60](trained-models/bach60)

[bach-4part-mxl](trained-models/bach-4part-mxl)

The models were trained with NVIDIA V100 of AI Bridging Cloud Infrastructure (ABCI) 
provided by National Institute of Advanced Industrial Science and Technology (AIST).

## Testing
NOTE that the cross validation set No. in `--model_to_test` and `--cv_set_no` must match when using `--dataset bach60`.
```
python run.py \
--do_test \
--device cpu \
--activation_fn tanh \
--seed 123 \
--num_modes 2 \
--inter_key_transition_limit 0.01 \
--key_preprocessing key-preprocess-none \
--dataset <dataset-name> \
--cv_set_no <targeted-cv-set-no> \
--model_to_test <targeted-model-filename>
```

## Generating Graphs of HSMM parameters
```
python run.py \
--make_hsmm_graphs \
--device cpu \
--activation_fn tanh \
--seed 123 \
--num_modes 2 \
--inter_key_transition_limit 0.01 \
--key_preprocessing key-preprocess-none \
--dataset <dataset-name> \
--cv_set_no <targeted-cv-set-no> \
--model_to_test <targeted-model-filename>
```
## Acknowledgments
This research has been supported by JSPS KAKENHI No. 23K20011.
Computational resource of AI Bridging Cloud Infrastructure (ABCI) 
provided by National Institute of Advanced Industrial Science and Technology (AIST) was used. 
