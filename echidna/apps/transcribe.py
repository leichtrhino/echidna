
import argparse

def attach_parser(parser):
    pass

def main(args):
    raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('echidna-transcribe')
    attach_parser(parser)
    args = parser.parse_args()
    main(args)
