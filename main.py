import os
from argparse import ArgumentParser

import src.detect as detect


if __name__ == "__main__":
    parser = ArgumentParser(description='Deep End-to-End Calibration')
    parser.add_argument("--mode", type=str, default="train", help="operation to perform on model")
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs run (default: 200)")
    parser.add_argument("--batch_sizes", type=tuple, default=(12, 12, 12),
                        help="batch size for (training, validation, test)")
    parser.add_argument("--data_dir", type=str, help="location of data", default="data")
    parser.add_argument("--log_dir", type=str, help="location of logging", default="log")
    parser.add_argument("--fractions", type=tuple, help="how much of the data to use for (training, validation, test)",
                        default=(0.88, 0.1, 0.02))
    parser.add_argument("--workers", type=int, help="number of data loading workers", default=2)
    parser.add_argument("--use_gpu", type=bool, help="whether to parallelise(default: True)", default=True)
    args = parser.parse_args()

    if args.mode == "train":
        # noise_samples = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.2]
        # for i, noise_sample in enumerate(noise_samples):
        #     detect.train(n_epochs=args.n_epochs, batch_sizes=args.batch_sizes, data_dir=args.data_dir,
        #                  log_dir=os.path.join(args.log_dir, "{}".format(noise_sample)),
        #                  fractions=args.fractions, workers=args.workers, use_gpu=args.use_gpu, standardize=False,
        #                  noise=noise_sample)
        detect.train(n_epochs=args.n_epochs, batch_sizes=args.batch_sizes, data_dir=args.data_dir,
                     log_dir=args.log_dir,
                     fractions=args.fractions, workers=args.workers, use_gpu=args.use_gpu, standardize=False,
                     noise=0.0)

    elif args.mode == "test":
        detect.test(batch_sizes=args.batch_sizes, data_dir=args.data_dir, log_dir=args.log_dir,
                    fractions=args.fractions, workers=args.workers, use_gpu=args.use_gpu, standardize=False)
