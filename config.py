d_config = {
    "data_path": "/Users/kaleempeeroo/PhD/Datasets/ML/combined_no_noise_sample.parquet",
    "parameter_columns": [
        "datalen_bytes",
        "pub_count",
        "sub_count",
        "use_reliable",
        "use_multicast",
        "durability",
    ],
    "metric_columns": ["latency_us"],
    "explore_classifications": {
        "enabled": True,
        "metric": "latency_us",
        "section": {"start": 0, "end": 1},
    },
    "undesired_datasets": [],
    "sections": [
        {"name": "0_to_5", "start": 0, "end": 0.05},
        {"name": "5_to_95", "start": 0.05, "end": 0.95},
        {"name": "95_to_100", "start": 0.95, "end": 1},
        {"name": "0_to_100", "start": 0, "end": 1},
    ],
    "overwrite_processing": {
        "variations": False,
        "all_pairings": False,
        "most_perf_pairings": False,
    },
    "dev_mode": True,
    "dev_config": {
        "varied_param_list": ["datalen_bytes"],
        "filter_option": 1,
        "explore_classifications": True,
    },
}
