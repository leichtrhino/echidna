
import argparse
import logging

from . import apps

parser = argparse.ArgumentParser('echidna',
                                 description='description')
subparsers = parser.add_subparsers(title='subcommands',
                                   description='description',
                                   help='additional help',
                                   dest='subcommand')
data_parser = subparsers.add_parser('data')
data_subparsers = data_parser.add_subparsers(title='datacommands',
                                             description='description',
                                             help='additional help',
                                             dest='datasubcommand')

apps.data_samples.attach_parser(data_subparsers.add_parser('samples'))
apps.data_augmentations.attach_parser(data_subparsers.add_parser('augmentations'))
apps.data_mixtures.attach_parser(data_subparsers.add_parser('mixtures'))
apps.train.attach_parser(subparsers.add_parser('train'))
apps.validate.attach_parser(subparsers.add_parser('validate'))
apps.separate.attach_parser(subparsers.add_parser('separate'))
apps.cluster.attach_parser(subparsers.add_parser('cluster'))
apps.transcribe.attach_parser(subparsers.add_parser('transcribe'))

args = parser.parse_args()

if args.subcommand == 'data':
    if args.datasubcommand == 'samples':
        apps.data_samples.main(args)
    elif args.datasubcommand == 'augmentations':
        apps.data_augmentations.main(args)
    elif args.datasubcommand == 'mixtures':
        apps.data_mixtures.main(args)
elif args.subcommand == 'train':
    apps.train.main(args)
elif args.subcommand == 'validate':
    apps.validate.main(args)
elif args.subcommand == 'separate':
    apps.separate.main(args)
elif args.subcommand == 'cluster':
    apps.cluster.main(args)
elif args.subcommand == 'transcribe':
    apps.transcribe.main(args)
