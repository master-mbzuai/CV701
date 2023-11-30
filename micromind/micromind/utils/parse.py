import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="General configuration for micromind.")

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument(
        "--optimizer",
        dest="opt",
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer name.",
    )
    parser.add_argument(
        "--experiment_name", default="pca", help="Name of the experiment."
    )
    parser.add_argument(
        "--output_folder", default="results", help="Output folder path."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode to check train and validation steps.",
    )
    parser.add_argument(
        "--d",
        default="0",
        type=int,
        )
    parser.add_argument(
        "--model_name",
        default="original_pre_pca",
        )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int
    )

    args = parser.parse_args()
    return args
