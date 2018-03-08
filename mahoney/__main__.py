import argparse

import mahoney
from mahoney.exp_nmf import std_nmf, property_nmf

def info(args):
    '''Print system info.
    '''
    import sys
    print('Python version:', sys.version)


def main():
    parser = argparse.ArgumentParser(
        description='Distributed neuron segmentation',
        argument_default=argparse.SUPPRESS,
    )
    subcommands = parser.add_subparsers()

    # info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    cmd = subcommands.add_parser('std_nmf', description='run baseline experiment')
    cmd.set_defaults(func=std_nmf)

    cmd = subcommands.add_parser('property_nmf', description='run baseline experiment with erode +dilate')
    cmd.set_defaults(func=property_nmf)

    # Each subcommand gives an `args.func`.
    # Call that function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
