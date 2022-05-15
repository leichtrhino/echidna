# sample evaluation
python scripts/03.eval-separator.py \
  --validation-dir \
    "/mnt/DC6ABAFE6ABAD48C/Datasets/Echidna/Echidna-speech-enhancement-train" \
  --output scores-dmx-050.json \
  --n-fft 2048 \
  --hop-length 512 \
  --batch-size 6 \
  --gpu \
  --loss source-to-distortion-ratio multiscale-spectrum \
  --loss-weight 1.0 1.0 \
  --checkpoint "/mnt/DC6ABAFE6ABAD48C/audio-source-separation/dmx-050.tar" \
  --log-level INFO \
  --log evaluation.log

python scripts/03.eval-separator.py \
  --validation-dir \
    "/mnt/DC6ABAFE6ABAD48C/Datasets/Echidna/Echidna-speech-enhancement-extra-train" \
  --output scores-dmx-050.json \
  --n-fft 2048 \
  --hop-length 512 \
  --batch-size 6 \
  --gpu \
  --loss source-to-distortion-ratio multiscale-spectrum \
  --loss-weight 1.0 1.0 \
  --checkpoint "/mnt/DC6ABAFE6ABAD48C/audio-source-separation/dmx-050.tar" \
  --log-level INFO \
  --log evaluation.log

python scripts/03.eval-separator.py \
  --validation-dir \
    "/mnt/DC6ABAFE6ABAD48C/Datasets/Echidna/Echidna-speech-enhancement-synth-train" \
  --output scores-dmx-050.json \
  --n-fft 2048 \
  --hop-length 512 \
  --batch-size 6 \
  --gpu \
  --loss source-to-distortion-ratio multiscale-spectrum \
  --loss-weight 1.0 1.0 \
  --checkpoint "/mnt/DC6ABAFE6ABAD48C/audio-source-separation/dmx-050.tar" \
  --log-level INFO \
  --log evaluation.log

python scripts/03.eval-separator.py \
  --validation-dir \
    "/mnt/DC6ABAFE6ABAD48C/Datasets/Echidna/Echidna-vocal-enhancement-train" \
  --output scores-dmx-050.json \
  --n-fft 2048 \
  --hop-length 512 \
  --batch-size 6 \
  --gpu \
  --loss source-to-distortion-ratio multiscale-spectrum \
  --loss-weight 1.0 1.0 \
  --checkpoint "/mnt/DC6ABAFE6ABAD48C/audio-source-separation/dmx-050.tar" \
  --log-level INFO \
  --log evaluation.log

