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
    pretrain_weight = "../datas/output/Benetech_AllChartTrain_v2/20230611143020"
    target_chart_type = "vertical_bar"
    train_csv_list = [
        {
            "csv_path": "../datas/input/benetech-making-graphs-accessible/extracted_clean_4fold.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [1,2,3],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/ICDAR_2022/useful/cleaned_fold_4.csv",
            "data_path": "../datas/input/ICDAR_2022/useful",
            "train_folds": [],
            "val_folds": [0]
        },

        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_default/gnuplot_gen_default_vertical_bar_8fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_default",
            "train_folds": [0,1,2,3],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_default_include_zero/gnuplot_gen_default_include_zero_vertical_bar_4fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_default_include_zero",
            "train_folds": [0,1],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_histogram_v2/gnuplot_gen_histogram_v2_4fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_histogram_v2",
            "train_folds": [0,1],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_longangle_include_zero/gnuplot_gen_longangle_vertical_bar_include_zero_6fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_longangle_include_zero",
            "train_folds": [0,1,2],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_longangle_v2/gnuplot_gen_longangle_vertical_bar_v2_6fold.csv",
            "data_path": "../datas/input/generated_synthetic/vertical_bar/finetuning/data_longangle_v2",
            "train_folds": [0,1,2],
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
    output_path = "../datas/output/Benetech_SpecificChartTrain_VerticalBar_1st"
    log_steps = 200
    batch_size = 2
    val_batch_size = 4
    use_wandb = False
    