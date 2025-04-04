import argparse
import os

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing all document files to merge",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    group = parser.add_argument_group(title="miscellaneous")
    group.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the datasets are assumed to be multimodal"
    )

    args = parser.parse_args()

    assert os.path.isdir(
        args.input
    ), f"ERROR: {args.input} is not a directory or does not exist"

    assert os.path.isdir(
        os.path.dirname(args.output_prefix)
    ), f"ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist"

    return args


def main(args):
    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(
            os.path.join(args.input, prefix) + ext_pair
        ), f"ERROR: {ext_pair} file not provided for {os.path.join(args.input, prefix)}"

        prefixes.add(prefix)

    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            dataset = IndexedDataset(os.path.join(args.input, prefix), multimodal=args.multimodal)
            builder = IndexedDatasetBuilder(
                get_bin_path(args.output_prefix), dtype=dataset.index.dtype, multimodal=args.multimodal
            )
            del dataset

        builder.add_index(os.path.join(args.input, prefix))

    builder.finalize(get_idx_path(args.output_prefix))


if __name__ == '__main__':
    args_ = get_args()
    main(args_)