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
            "csv_path": "../datas/input/benetech-making-graphs-accessible/extracted_4fold_axis_reading.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [1,2,3],
            "val_folds": [0]
        },
        {
            "csv_path": "../datas/input/benetech-making-graphs-accessible/generated_4fold_axis_reading.csv",
            "data_path": "../datas/input/benetech-making-graphs-accessible/train",
            "train_folds": [0,1,2,3],
            "val_folds": []
        },
        {
            "csv_path": "../datas/input/ICDAR_2022/useful/fold_4_axis_reading.csv",
            "data_path": "../datas/input/ICDAR_2022/useful",
            "train_folds": [1,2,3],
            "val_folds": [0]
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
    output_path = "../datas/output/AxisReadning"
    log_steps = 200
    batch_size = 2
    val_batch_size = 4
    use_wandb = False
    