#!/usr/env python
import argparse
import csv
import logging

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from echidna import datasets as ds

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-n', '--samples', type=int, default=1000)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-p', '--process', type=int, default=None)
    parser.add_argument('--log')
    parser.add_argument('--log-level', default='INFO')

    # parameters for column names of csv
    parser.add_argument('--path-colname', default='path')
    parser.add_argument('--category-colname', default='category')
    parser.add_argument('--track-colname', default='track')
    parser.add_argument('--split-colname', default='split')

    # categories and tracks and split
    parser.add_argument('--num-sources', type=int)
    parser.add_argument('--categories', nargs='*')
    parser.add_argument('--splits', nargs='*')
    parser.add_argument('--weights', type=float, nargs='*')
    parser.add_argument('--ignore-tracks', action='store_true')
    parser.add_argument('--force-order', action='store_true')
    parser.add_argument('--allow-category-repetition', action='store_true')

    # parameters for load
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--duration', type=float, default=5.0)

    # augmentation
    parser.add_argument('--scale-ratio', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--scale-point', type=int, default=2)
    parser.add_argument('--shift-ratio', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--stretch-ratio', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--normalize-scale', action='store_true')

    # mix algorithm
    parser.add_argument('--mix-name', type=str, required=True)
    parser.add_argument('--mix-category', nargs='+', action='append')
    parser.add_argument('--mix-include-other', action='store_true')

    args = parser.parse_args()

    # validation
    if not args.num_sources and not args.categories:
        raise ValueError(
            'at least one of --num-sources and --categories must be given')

    if args.num_sources and args.categories and args.force_order:
        raise ValueError(
            'giving all of --num-sources and --categories and --force-order is invalid')

    if args.weights is not None:
        if not args.categories:
            raise ValueError(
                '--categories should be given if --weights is given')
        args.weights = dict(zip(args.categories, args.weights))

    if args.force_order:
        args.num_sources = None
        args.categories = list(args.categories)
    else:
        if args.num_sources is None:
            args.num_sources = len(args.categories)
        if args.categories is not None:
            args.categories = set(args.categories)

    return args

def main():
    args = parse_args()
    if args.log is not None:
        logging.basicConfig(filename=args.log, level=args.log_level)
    else:
        logging.basicConfig(stream=sys.stdout, level=args.log_level)

    # load csv file
    with open(args.input, 'r') as fp:
        source_list_reader = csv.reader(fp)
        cols = next(source_list_reader)
        path_index = cols.index(args.path_colname)
        if path_index < 0:
            raise ValueError(
                f'column {args.path_colname} not found in {args.input}')
        category_index = cols.index(args.category_colname)
        if category_index < 0:
            raise ValueError(
                f'column {args.category_colname} not found in {args.input}')
        track_index = -1
        if not args.ignore_tracks:
            track_index = cols.index(args.track_colname)
            if track_index < 0:
                raise ValueError(
                    f'column {args.track_colname} not found in {args.input}')
        split_index = -1
        if args.splits:
            split_index = cols.index(args.split_colname)
            if split_index < 0:
                raise ValueError(
                    f'column {args.split_colname} not found in {args.input}')

        source_list = dict((
            l[path_index],
            {
                'category': l[category_index],
                'track': l[track_index] if track_index >= 0 else None,
                'split': l[split_index] if split_index >= 0 else None,
            }
        ) for l in source_list_reader)

    if not os.path.exists(args.output):
        # build Sampler
        sampler = ds.Sampler(
            source_list,
            sr=args.sr,
            duration=args.duration,
            source_categories=args.categories,
            num_sources=args.num_sources,
            category_repetition=args.allow_category_repetition,
            category_weight=args.weights,
            splits=args.splits,
            check_track_strictly=False) \
                    .scale_range(args.scale_ratio) \
                    .scale_point_range((2, args.scale_point)) \
                    .pitch_shift_range(args.shift_ratio) \
                    .time_stretch_range(args.stretch_ratio) \
                    .normalize(args.normalize_scale) \
                    .trial_num(10000)

        # freeze Sampler
        sampler.freeze(
            out_dir=args.output,
            num_samples=args.samples,
            num_process=args.process)

    # calculate mix
    ds.freeze_mix(
        sampler=args.output,
        algorithm=ds.CategoryMix(
            args.mix_category,
            output_other=args.mix_include_other),
        name=args.mix_name)

if __name__ == '__main__':
    main()

