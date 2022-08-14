import logging

try:
    import cudf
    import cupy as cp
except:
    import numpy as cp
    import pandas as cudf

logger = logging.getLogger("MetaBCC-LR")


def sample(composition_file, coverage_file, fraction, output):
    composition = cudf.read_csv(composition_file, header=None)
    coverage = cudf.read_csv(coverage_file, header=None)
    no_rows, _ = coverage.shape
    sampling_size = int(no_rows * fraction)

    logger.debug(f"Composition data shape {composition.shape}")
    logger.debug(f"Coverage data shape {coverage.shape}")
    logger.debug(f"Sampling {sampling_size} from {no_rows} data points")

    row_indices = cp.random.choice(no_rows, sampling_size, replace=False)
    composition_sampled = composition.iloc[row_indices]
    coverage_sampled = coverage.iloc[row_indices]

    composition_sampled.to_csv(
        f"{output}/features/composition_sampled_{fraction}.csv",
        index=False,
        header=None,
    )
    coverage_sampled.to_csv(
        f"{output}/features/coverage_sampled_{fraction}.csv", index=False, header=None
    )
    cudf.DataFrame(row_indices, columns=["Sequence Index"]).to_csv(
        f"{output}/features/indices_{fraction}.csv", index=False, header=None
    )

    return composition_sampled, coverage_sampled, row_indices
