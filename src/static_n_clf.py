"""
Static Node Classification
"""

from rf_lr_clf import *
from mlp_binary_clf import *


def main():
    """
    static node classification
    :return:
    """
    # GENERAL arguments
    parser = argparse.ArgumentParser(description='Static Node Classification.')
    parser.add_argument('--network', type=str, default='otc', help='Network name')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of iteration.')
    parser.add_argument('--clf', type=str, default='RF', help='Name of classifier.')

    # MLP specific arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of runs.')
    parser.add_argument('--bs', type=int, default=100, help='Batch size.')
    parser.add_argument('--drop_out', type=float, default=0.1, help='drop-out ratio.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rage.')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epoch.')
    parser.add_argument('--h_1', type=int, default=80, help='Hidden layer 1 dimension.')
    parser.add_argument('--h_2', type=int, default=10, help='Hidden layer 2 dimension.')

    try:
        args = parser.parse_args()
        print("Arguments:", args)
    except:
        parser.print_help()
        sys.exit()

    classifier = args.clf
    if classifier == 'MLP':
        end_to_end_n_clf_with_MLP(args)
    elif classifier == 'RF' or classifier == 'LR':
        end_to_end_rf_lr_clf(args)
    else:
        raise ValueError("Undefined classifier!")


if __name__ == '__main__':
    main()