"SVM": {
            "SST2_1_c": {"train_time": None, "predict_time": None},
            "SST2_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_r": {"train_time": None, "predict_time": None},
        },

models = {
    "Machine Learning": {
        "SVM": {
            "SST2_1_c": {"train_time": None, "predict_time": None},
            "SST2_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_r": {"train_time": None, "predict_time": None},
        },
        "NB": {
            "SST2_1_c": {"train_time": None, "predict_time": None},
            "SST2_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_r": {"train_time": None, "predict_time": None},
    },
    "Deep Learning": {
        "MLP": {
            "SST2_1_c": {"train_time": 19.04027049, "predict_time": 1.043880542},
            "SST2_2_c": {"train_time": 3.531718254, "predict_time": 0.20336771},
            "SST5_2_c": {"train_time": 3.508548498, "predict_time": 0.198204597},
            "SST5_2_r": {"train_time": 4.433998982, "predict_time": 0.21322751},
        },
        "CNN": {
            "SST2_1_c": {"train_time": 22.34779223, "predict_time": 1.094512145},
            "SST2_2_c": {"train_time": 3.977898121, "predict_time": 0.256754796},
            "SST5_2_c": {"train_time": 3.883461793, "predict_time": 0.225681384},
            "SST5_2_r": {"train_time": 5.14701883, "predict_time": 0.247698069},
        },
        "RNN": {
            "SST2_1_c": {"train_time": 1483.427321, "predict_time": 19.66347718},
            "SST2_2_c": {"train_time": 197.2718136, "predict_time": 2.751911322},
            "SST5_2_c": {"train_time": 202.5867754, "predict_time": 2.814872583},
            "SST5_2_r": {"train_time": 193.5351152, "predict_time": 2.713918527},
        },
        "LSTM": {
            "SST2_1_c": {"train_time": 93.82405019, "predict_time": None},
            "SST2_2_c": {"train_time": 16.49923897, "predict_time": None},
            "SST5_2_c": {"train_time": 16.94786135, "predict_time": None},
            "SST5_2_r": {"train_time": 16.64784638, "predict_time": 1.545251369},
        },
        "GRU": {
            "SST2_1_c": {"train_time": 89.65073625, "predict_time": None},
            "SST2_2_c": {"train_time": 15.87023671, "predict_time": None},
            "SST5_2_c": {"train_time": 15.9026413, "predict_time": None},
            "SST5_2_r": {"train_time": 15.95740724, "predict_time": 1.406955083},
        },



    },
    "Transformer": {
        "BERT": {
            "SST2_1_c": {"train_time": None, "predict_time": None},
            "SST2_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_r": {"train_time": None, "predict_time": None},
        },
        # Add more transformer models as needed
    },
    "Lexicon": {
            "SST2_1_c": {"train_time": None, "predict_time": None},
            "SST2_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_c": {"train_time": None, "predict_time": None},
            "SST5_2_r": {"train_time": None, "predict_time": None},
    },
    # Add more categories as needed
}
