"""Verify and compare five FastWAM models under the official one-trial protocol."""

import aggregate_fastwam_libero_results as common

# The upstream project documents one trial for the 10,030-task LIBERO-plus catalog.
common.EPISODES_PER_TASK = 1

import aggregate_fastwam_libero_five_model_results as five_model


if __name__ == "__main__":
    five_model.main()
