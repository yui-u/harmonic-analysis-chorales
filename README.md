# Automatic Mode Recognition and Harmonic Analysis with Unlabeled Data
This repository is the implementation of 
["Unsupervised Learning of Harmonic Analysis Based on Neural HSMM with Code Quality Templates"](http://arxiv.org/abs/2403.04135)
(Uehara; Proceedings of the 11th International Conference on New Music Concepts, pp.30-53) and the extended paper ["Automatic Mode Recognition and Harmonic Analysis with Unlabeled Data"](https://doi.org/10.48293/IJMSTA-115) (Uehara; International Journal of Music Science, Technology and Art. Vol. 6. Issue 2, pp.17-37).

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

## Training with a fixed number of modes
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
- `--resolution_str` can be selected from \{16th, half-beat, beat\} (half-beat is recommended).
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
--resolution_str <resolution_str> \
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
--resolution_str <resolution_str> \
--dataset bach-4part-mxl \
--batch_size 8 \
--model_to_initialize <the-model-trained-in-phase1>
```

## Training with a dynamic number of modes
### Training Phase1
(Pre-)Training the model with the normalized data and the no_shift setting.
If GPU is not available, change to `--device cpu`.
- To perform training with a dynamic number of modes, set the `--dynamic_num_modes` flag.
- `--num_modes` should be set to 1.
- `--acceptance_th` is a threshold that requests a certain improvement in the marginal likelihood obtained by increasing the number of modes.
- `--cossim_limit` is a threshold for requesting that a newly added mode have a distribution different from the previously accepted modes in terms of the stationary distribution.

#### bach-4part-mxl
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123 \
--num_modes 1 \
--inter_key_transition_limit 0.0 \
--no_shift \
--num_epochs 480 \
--key_preprocessing key-preprocess-normalized \
--resolution_str half-beat \
--dynamic_num_modes \
--warmup_num_modes 16 \
--acceptance_th 1e-2 \
--cossim_limit 0.8 \
--dataset bach-4part-mxl \
--batch_size 8
```

### Training Phase2
Additional training with original data without normalization.
Shift and inter_key_transition are enabled.
In the Phase 2 training, the model learned in Phase 1 is used as the initial value.
- The `--dynamic_num_modes` flag should be set and `--num_modes` should again be set to 1.
- The `--acceptance_th` and `--cossim_limit` are the same as the Phase 1 settings.

#### bach-4part-mxl
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123 \
--num_modes 1 \
--inter_key_transition_limit 0.01 \
--num_epochs 240 \
--key_preprocessing key-preprocess-none \
--resolution_str half-beat \
--dynamic_num_modes \
--warmup_num_modes 16 \
--acceptance_th 1e-2 \
--cossim_limit 0.8 \
--dataset bach-4part-mxl \
--batch_size 8 \
--model_to_initialize <the-model-trained-in-phase1>
```

### Finetuning
Finetuning with the fixed (learned) number of modes.
- Set `--num_modes` to the value obtained from the phase 2 training.
- The `--dynamic_num_modes` flag is NOT set.

#### bach-4part-mxl
```
python run.py \
--do_train \
--device cuda:0 \
--dir_instance_output <output-dir-name> \
--activation_fn tanh \
--seed 123 \
--num_modes <num-modes> \
--inter_key_transition_limit 0.01 \
--num_epochs 240 \
--key_preprocessing key-preprocess-none \
--resolution_str half-beat \
--dataset bach-4part-mxl \
--batch_size 8 \
--model_to_initialize <the-model-trained-in-phase2>
```

## Trained models
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
--inter_key_transition_limit 0.01 \
--key_preprocessing key-preprocess-none \
--resolution_str <resolution_str> \
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

## Publications
Please cite the following paper when using this code:
```
Uehara, Y. (2024). Automatic Mode Recognition and Harmonic Analysis with Unlabeled Data. 
International Journal of Music Science, Technology and Art (IJMSTA). 2024 July 01; 6 (2): 17-37. 
```
