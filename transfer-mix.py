
import os
import json
import math

dataset_dirs = ['dataset']
filter_fraction = 0.4

def process_dir(dataset_dir):
    # load partition_metadata.json
    with open(os.path.join(dataset_dir, 'partition_metadata.json'), 'r') as fp:
        partition_metadata = json.load(fp)

    # load scores-dmx-100.json
    with open(os.path.join(dataset_dir, 'scores-dmx-100.json'), 'r') as fp:
        score = [s['loss'] for s in json.load(fp)]

    # load source_metadata.json
    with open(os.path.join(dataset_dir, 'source_metadata.json'), 'r') as fp:
        source_metadata = json.load(fp)

    if not os.path.exists(os.path.join(dataset_dir, 'mix')):
        os.mkdir(os.path.join(dataset_dir, 'mix'))
    # output partition_metadata to mix/voice-base
    with open(os.path.join(dataset_dir, 'mix', 'voice-base'), 'w') as fp:
        json.dump(partition_metadata, fp)

    # find target score
    sorted_score = sorted(score, reverse=True)
    samples = int(filter_fraction * len(sorted_score))
    sorted_score.insert(0, math.inf)
    target_score = sorted_score[samples]

    # zip partition and score
    sorted_mix = sorted(partition_metadata.items())
    filtered_mix = dict()
    last_idx = 0
    for p, m in sorted_mix:
        f_mix = [
            _m for _m, s in zip(m, score[last_idx:last_idx+len(m)])
            if s > target_score
        ]
        last_idx += len(m)
        if f_mix:
            filtered_mix[p] = f_mix

    with open(os.path.join(dataset_dir, 'mix', 'voice-dmx100-50'), 'w') as fp:
        json.dump(filtered_mix, fp)

def main():
    for dataset_dir in dataset_dirs:
        process_dir(dataset_dir)

if __name__ == '__main__':
    main()
