d_config = {
    "data_path": "/Users/kaleempeeroo/PhD/Datasets/ML/3PI_No_Noise_PCG_100B_1KB.parquet",
    "parameter_columns": [
        "datalen_bytes",
        "pub_count",
        "sub_count",
        "use_reliable",
        "use_multicast",
        "durability",
    ],
    "metric_columns": ["latency_us"],
    "undesired_datasets": [
        "3PI ML 6Mbps",
        "3PI ML 8Mbps",
        "5PI ML 24Mbps",
        "5PI ML 71Mbps",
        "5PI Mcast 1P3S BW High",
        "5PI Mcast 1P3S BW Medium",
    ],
    "sections": [
        {"name": "0_to_5", "start": 0, "end": 0.05},
        {"name": "5_to_95", "start": 0.05, "end": 0.95},
        {"name": "95_to_100", "start": 0.95, "end": 1},
        {"name": "0_to_100", "start": 0, "end": 1},
    ],
    "overwrite_processing": {
        "variations": True,
        "all_pairings": True,
        "most_perf_pairings": True,
    },
    "dev_mode": True,
    "dev_config": {
        "varied_param_list": ["datalen_bytes"],
        "filter_option": 1,
        "explore_classifications": True,
    },
}
