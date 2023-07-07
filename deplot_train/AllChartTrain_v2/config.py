class CFG:

    # General
    debug = False
    num_proc = 1
    num_workers = 0
    gpus = 1

    # Data
    max_length = 512
    image_height = 840
    image_width = 840

    # Training
    train_csv_list = [
        {
            "csv_path": "../datas/input/benetech-making-graphs-accessible/extracted_clean_4fold.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [1,2,3],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/benetech-making-graphs-accessible/generated_clean_4fold.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [0,1,2,3],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/ICDAR_2022/useful/fold_4.csv",
            "data_path": "../datas/input/ICDAR_2022/useful",
            "train_folds": [],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/generated_synthetic/dot/data_categorical_v2/data_list_category_v2.csv",
            "data_path": "../datas/input/generated_synthetic/dot/data_categorical_v2",
            "train_folds": [0],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/dot/data_numerical/numerical_10fold.csv",
            "data_path": "../datas/input/generated_synthetic/dot/data_numerical/",
            "train_folds": [1,2,3,4,5,6,7,8,9],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/horizontal_bar/data_Bartley/data_list.csv",
            "data_path": "../datas/input/generated_synthetic/horizontal_bar/data_Bartley",
            "train_folds": [0],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/histogram/gnuplot_gen_histogram_10fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/histogram",
            "train_folds": [1,2,3,4,5,6,7,8,9],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/generated_synthetic/line/data_algebraicline/gnuplot_gen_algebraicline_10fold.csv",
            "data_path": "../datas/input/generated_synthetic/line/data_algebraicline",
            "train_folds": [0,1,2,3,4,5,6,7,8,9],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/line/data_log/gnuplot_gen_logline_10fold.csv",
            "data_path": "../datas/input/generated_synthetic/line/data_log",
            "train_folds": [0,1,2,3,4,5,6,7,8,9],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/scatter/data_v1/gnuplot_gen_scatter_10fold.csv",
            "data_path": "../datas/input/generated_synthetic/scatter/data_v1",
            "train_folds": [0,1,2,3,4,5,6,7,8,9],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/scatter/data_v2/gnuplot_gen_scatter_10fold_v2_ok.csv",
            "data_path": "../datas/input/generated_synthetic/scatter/data_v2",
            "train_folds": [0,1,2,3,4,5,6,7,8,9],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/horizontal_bar/data_long/gnuplot_gen_long_horizontal_bar_5fold.csv",
            "data_path": "../datas/input/generated_synthetic/horizontal_bar/data_long",
            "train_folds": [0,1,2,3,4],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/horizontal_bar/data_newline/gnuplot_gen_newline_horizontal_bar_5fold.csv",
            "data_path": "../datas/input/generated_synthetic/horizontal_bar/data_newline",
            "train_folds": [0,1,2,3,4],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/data_longangle/gnuplot_gen_longangle_vertical_bar_5fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/data_longangle",
            "train_folds": [0,1,2,3,4],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/data_newline/gnuplot_gen_newline_vertical_bar_5fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/data_newline",
            "train_folds": [0,1,2,3,4],
            "val_folds": []
        },
    ]
    fold = 0
    epochs = 8
    val_check_interval = 1.0  # how many times we want to validate during an epoch
    check_val_every_n_epoch = 1
    gradient_clip_val = 1.0
    lr = 1e-5
    lr_scheduler_type = "cosine"
    num_warmup_steps = 100
    seed = 42
    warmup_steps = 300  
    output_path = "../datas/output/Benetech_AllChartTrain_v2"
    log_steps = 200
    batch_size = 2
    val_batch_size = 4
    use_wandb = False
    