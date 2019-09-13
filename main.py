import os
from argparse import ArgumentParser

import src.detect as detect


if __name__ == "__main__":
    parser = ArgumentParser(description='Deep End-to-End Calibration')
    parser.add_argument("--mode", type=str, default="test", help="operation to perform on model")
    parser.add_argument("--n_epochs", type=int, default=4000, help="number of epochs run (default: 200)")
    parser.add_argument("--batch_sizes", type=tuple, default=(12, 12, 2),
                        help="batch size for (training, validation, test)")
    parser.add_argument("--data_dir", type=str, help="location of data", default="data")
    parser.add_argument("--log_dir", type=str, help="location of logging", default="log")
    parser.add_argument("--fractions", type=tuple, help="how much of the data to use for (training, validation, test)",
                        default=(0.82, 0.09, 0.09))
    parser.add_argument("--workers", type=int, help="number of data loading workers", default=2)
    parser.add_argument("--use_gpu", type=bool, help="whether to parallelise(default: True)", default=True)
    args = parser.parse_args()

    if args.mode == "train":
        detect.train(n_epochs=args.n_epochs, batch_sizes=args.batch_sizes, data_dir=args.data_dir, log_dir=args.log_dir,
                     fractions=args.fractions, workers=args.workers, use_gpu=args.use_gpu, standardize=False)

    elif args.mode == "test":
        detect.test(batch_sizes=args.batch_sizes, data_dir=args.data_dir, log_dir=args.log_dir,
                    fractions=args.fractions, workers=args.workers, use_gpu=args.use_gpu, standardize=False)


xfrm = torch.tensor([[0.0000, 0.0000, 1.5000, 0.7071, 0.7071, 0.1736, 0.9848, 0.8660, 0.5000],
                  [0.0000, 0.0000, 1.0000, 0.7071, 0.7071, 0.1736, 0.9848, 0.8660, 0.5000]])
cam_xfrm = world_2_cam_xfrm(xfrm)

tensor([[[-0.4924,  0.0868, -0.8660, -0.6892],
         [-0.4803,  0.8027,  0.3536, -0.3553],
         [ 0.7259,  0.5900, -0.3536, -0.0764],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[-0.4924,  0.0868, -0.8660, -0.1162],
         [-0.4803,  0.8027,  0.3536,  0.3313],
         [ 0.7259,  0.5900, -0.3536,  1.3574],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]])


xfrm = torch.tensor([[0., 0., 1.5, 0.81373504, 0.28989174, 0.26034719, 0.43129735],
                  [0., 0., 1.0, 0.81373504, 0.28989174, 0.26034719, 0.43129735]])

cam_xfrm = world_2_cam_xfrm(xfrm, "quaternion")
xfrm_mat = qtvec_to_transformation_matrix(cam_xfrm)
tensor([[[ 0.4924, -0.6738, -0.5510,  0.0000],
         [ 0.1736,  0.6964, -0.6964, -1.5000],
         [ 0.8529,  0.2472,  0.4599,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[ 0.4924, -0.6738, -0.5510,  0.0000],
         [ 0.1736,  0.6964, -0.6964, -1.0000],
         [ 0.8529,  0.2472,  0.4599,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]])