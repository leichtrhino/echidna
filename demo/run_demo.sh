# make samples
python -m echidna data samples -c demo/config/samples/track_training.yaml
python -m echidna data samples -c demo/config/samples/notrack_training.yaml
python -m echidna data samples -c demo/config/samples/track_validation.yaml
python -m echidna data samples -c demo/config/samples/notrack_validation.yaml

# make augmentations
python -m echidna data augmentations -c demo/config/augmentations/track_training_random.yaml
python -m echidna data augmentations -c demo/config/augmentations/track_training_easy.yaml
python -m echidna data augmentations -c demo/config/augmentations/track_training_hard.yaml
python -m echidna data augmentations -c demo/config/augmentations/notrack_training_random.yaml
python -m echidna data augmentations -c demo/config/augmentations/notrack_training_easy.yaml
python -m echidna data augmentations -c demo/config/augmentations/notrack_training_hard.yaml
python -m echidna data augmentations -c demo/config/augmentations/track_validation_random.yaml
python -m echidna data augmentations -c demo/config/augmentations/notrack_validation_random.yaml

# make mixtures
python -m echidna data mixtures -c demo/config/mixtures/track_training.yaml
python -m echidna data mixtures -c demo/config/mixtures/notrack_training.yaml
python -m echidna data mixtures -c demo/config/mixtures/track_validation.yaml
python -m echidna data mixtures -c demo/config/mixtures/notrack_validation.yaml

# train
python -m echidna train -c demo/config/trainings/baseline_encdec_easy.yaml
python -m echidna train -c demo/config/trainings/baseline_encdec_hard.yaml
python -m echidna train -c demo/config/trainings/baseline_chimera_easy.yaml
python -m echidna train -c demo/config/trainings/baseline_chimera_hard.yaml

# validate
python -m echidna validate -c demo/config/validations/baseline_encdec_05.yaml
python -m echidna validate -c demo/config/validations/baseline_encdec_10.yaml
python -m echidna validate -c demo/config/validations/baseline_chimera_05.yaml
python -m echidna validate -c demo/config/validations/baseline_chimera_10.yaml

