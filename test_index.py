import pytest
import os
from index import main


class TestIndex:
    def test_main(self):
        d_config = {
            "data_path": "./tests/combined_no_noise_sample.parquet",
            "metric_columns": ["latency_us"],
            "undesired_datasets": [],
            "sections": [
                {"name": "0_to_100", "start": 0, "end": 1},
            ],
        }
        ls_expected_paths = [
            "./output/parameter_analysis_tool/classification_explorer/datalen_bytes/",
            "./output/parameter_analysis_tool/datalen_bytes/pairings",
            "./output/parameter_analysis_tool/datalen_bytes/variations/",
            "./output/parameter_analysis_tool/datalen_bytes/working_out_latency_us/",
            "./output/input_parameter_distribution_1.pdf",
            "./output/input_parameter_distribution_1.png",
            "./output/input_parameter_distribution_2.pdf",
            "./output/input_parameter_distribution_2.png",
        ]
        main(d_config)

        assert all([os.path.exists(path) for path in ls_expected_paths])
