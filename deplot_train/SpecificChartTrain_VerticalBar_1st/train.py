import os
import json
import random
from typing import List, Dict, Union, Tuple, Any
from functools import partial

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from transformers import (
    Pix2StructForConditionalGeneration,
    AutoProcessor,
    Pix2StructConfig,
)
from datasets import Dataset
from datasets import Image as ds_img
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

from config import CFG
from metrics_utils import *
from augmentations import get_train_transforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### tokens
PROMPT_TOKEN = "<CHART_TYPE>"
SEPARATOR_TOKENS = [
    PROMPT_TOKEN,
]
LINE_TOKEN =  "<line>" 
VERTICAL_BAR_TOKEN = "<vertical_bar>"
HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
SCATTER_TOKEN = "<scatter>"
DOT_TOKEN = "<dot>"

CHART_TYPE_TOKENS = [
    LINE_TOKEN,
    VERTICAL_BAR_TOKEN,
    HORIZONTAL_BAR_TOKEN,
    SCATTER_TOKEN,
    DOT_TOKEN,
]
new_tokens = SEPARATOR_TOKENS + CHART_TYPE_TOKENS

### fix seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG.seed)

### Model configuration
config = Pix2StructConfig.from_pretrained(CFG.pretrain_weight)
config.max_length = CFG.max_length

### Preprocess Settings
processor = AutoProcessor.from_pretrained(CFG.pretrain_weight)
processor.image_processor.size = {
    "height": CFG.image_height,
    "width": CFG.image_width,
}
num_added = processor.tokenizer.add_tokens(["<one>"] + new_tokens)
print(num_added, "tokens added")

config.pad_token_id = processor.tokenizer.pad_token_id
config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([PROMPT_TOKEN])[0]
one_token_id = processor.tokenizer("<one>", add_special_tokens=False).input_ids[0]
unk_token_id = processor.tokenizer.unk_token_id

def replace_unk_tokens_with_one(example_ids: List[int], example_tokens: List[str], one_token_id:int, unk_token_id:int) -> List[int]:
    """
    Replace unknown tokens that represent "1" with the correct token id.

    Args:
        example_ids (list): List of token ids for a given example
        example_tokens (list): List of tokens for the same given example
        one_token_id (int): Token id for the "<one>" token
        unk_token_id (int): Token id for the unknown token

    Returns:
        list: The updated list of token ids with the correct token id for "1"
    """
    
    temp_ids = []
    for id_, token in zip(example_ids, example_tokens):
        if id_ == unk_token_id and token == "1":
            id_ = one_token_id
        temp_ids.append(id_)
    return temp_ids

def preprocess(examples: Dict[str, str], processor: AutoProcessor, CFG: CFG, transforms) -> Dict[str, Union[torch.Tensor, List[int], List[str]]]:
    """
    Preprocess the given examples.

    This function processes the input examples by tokenizing the texts, replacing
    any unknown tokens that represent "1" with the correct token id, and loading
    the images.

    Args:
        examples (dict): A dictionary containing ground truth texts, image paths, and ids
        processor: An object responsible for tokenizing texts and processing images
        CFG: A configuration object containing settings and hyperparameters

    Returns:
        dict: A dictionary containing preprocessed images, token ids, and ids
    """
    # pixel_values = []
    flattened_patches = []
    attention_masks = []
    texts = examples["ground_truth"]

    ids = processor.tokenizer(
        texts,
        add_special_tokens=False,
        max_length=CFG.max_length,
        padding=True,
        truncation=True,
    ).input_ids

    if isinstance(texts, str):
        texts = [texts]

    tokens = [processor.tokenizer.tokenize(text, add_special_tokens=False) for text in texts]
    
    one_token_id = processor.tokenizer("<one>", add_special_tokens=False).input_ids[0]
    unk_token_id = processor.tokenizer.unk_token_id
    
    final_ids = [
        replace_unk_tokens_with_one(example_ids, example_tokens, one_token_id, unk_token_id)
        for example_ids, example_tokens in zip(ids, tokens)
    ]

    for sample in examples["image_path"]:
        arr = np.array(sample)
        if len(arr.shape) == 2:
            arr = np.stack([arr]*3, axis=-1)
        
        # Resize while preserving asect ratio
        height, width, channel = arr.shape
        if channel==4:
            arr = arr[:,:,0:3]
        if transforms is not None:
            arr = transforms(image=arr)["image"]
        long_edge = max(height, width)
        pad = (long_edge-min(height, width))//2
        resize_arr = np.zeros((long_edge, long_edge, 3), dtype=np.uint8)
        if height > width:
            resize_arr[0:height, pad:pad+width] = arr
        else:
            resize_arr[pad:pad+height, 0:width] = arr
        sample = Image.fromarray(resize_arr)

        encoded_image = processor(images=sample, text="Generate underlying data table of the figure below:", return_tensors="pt")
        flattened_patches.append(encoded_image.flattened_patches)
        attention_masks.append(encoded_image.attention_mask)

    return {
        "flattened_patches": flattened_patches,
        "attention_masks": attention_masks,
        "input_ids": final_ids,
        "id": examples["id"],
    }

### Dataset
def round_float(value):
    if isinstance(value, float):
        value = str(value)
        if "." in value:
            integer, dicimal = value.split(".")
            if abs(float(integer)) > 1:
                dicimal = dicimal[:1]
            else:
                dicimal = dicimal[:4]
            value = integer + "." + dicimal
        value = float(value)
    return value

def gen_data(df):
    id_list = df["id"].values
    gt_x_list = df["gt_x"].values
    gt_y_list = df["gt_y"].values
    chart_type_list = df["chart_type"].values
    image_path_list = df["image_path"].values

    for id, gt_x, gt_y, chart_type, image_path in zip(id_list, gt_x_list, gt_y_list, chart_type_list, image_path_list):
        nan_flag = False
        if "nan" in gt_y:
            gt_y = gt_y.replace("nan", "0")
            nan_flag = True

        all_x = eval(gt_x)
        all_y = eval(gt_y)

        if nan_flag:
            all_y[-1] = "NaN"

        if chart_type == "horizontal_bar":
            all_x = eval(gt_y)
            all_y = eval(gt_x)

        all_x = [round_float(x) for x in all_x]
        all_y = [round_float(x) for x in all_y]

        x_len = len(all_x)
        y_len = len(all_y)
        if x_len > y_len:
            all_y += [0]*(x_len-y_len)
        elif y_len > x_len:
            all_x += [0]*(y_len-x_len)
        data_series = []
        for x, y in zip(all_x, all_y):
            data_series.append(str(x)+" | "+str(y))
        data_series_str = " <0x0A> ".join(data_series)

        gt_string =  f" <0x0A> " + data_series_str + "</s>"

        yield {
            "ground_truth": gt_string,
            "x": json.dumps(all_x),
            "y": json.dumps(all_y),
            "chart-type": chart_type,
            "id": id,
            "image_path": os.path.join(image_path, f"{id}.jpg"),
        }

train_df_list = []
val_df_list = []
for train_csv_info in CFG.train_csv_list:
    df = pd.read_csv(train_csv_info["csv_path"])
    df["image_path"] = os.path.join(train_csv_info["data_path"], "images")

    target_chart_type = CFG.target_chart_type
    train_folds = train_csv_info["train_folds"]
    train_df = df.query("fold in @train_folds and chart_type==@target_chart_type")
    val_folds = train_csv_info["val_folds"]
    valid_df = df.query("fold in @val_folds and chart_type==@target_chart_type")
    if len(train_df) > 0:
        train_df_list.append(train_df)
        print(f'Add Train -> {train_csv_info["csv_path"]}/fold:{train_folds}')
    if len(valid_df) > 0:
        val_df_list.append(valid_df)
        print(f'Add Valid -> {train_csv_info["csv_path"]}/fold:{val_folds}')

train_df = pd.concat(train_df_list).reset_index(drop=True)
val_df = pd.concat(val_df_list).reset_index(drop=True)
print(f"all_data: {len(train_df)+len(val_df)}, train_data:{len(train_df)}/val_data:{len(val_df)}")

train_ds = Dataset.from_generator(
    gen_data, gen_kwargs={"df": train_df}, num_proc=CFG.num_proc
)
train_ds = train_ds.cast_column("image_path", ds_img())
train_transforms = get_train_transforms()
train_ds.set_transform(partial(preprocess, processor=processor, CFG=CFG, transforms=train_transforms))

# Creat validation set from only extracted examples
val_gt_ds = Dataset.from_generator(
    gen_data, gen_kwargs={"df": val_df}, num_proc=CFG.num_proc
)
val_ds = val_gt_ds.cast_column("image_path", ds_img())
val_ds.set_transform(partial(preprocess, processor=processor, CFG=CFG, transforms=None))

gt_chart_type = val_gt_ds["chart-type"]
gt_x = [json.loads(_) for _ in val_gt_ds["x"]]
gt_y = [json.loads(_) for _ in val_gt_ds["y"]]
gt_ids = val_gt_ds["id"]


### Dataloaders
pad_token_id = processor.tokenizer.pad_token_id
def collate_fn(samples: List[Dict[str, Union[torch.Tensor, List[int], str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for DataLoader.

    This function takes a list of samples and combines them into a batch with
    properly padded input_ids.

    Args:
        samples (List[Dict[str, Union[torch.Tensor, List[int], str]]]): 
            A list of samples, where each sample is a dictionary containing
            "pixel_values" (torch.Tensor), "input_ids" (List[int]), and "id" (str).

    Returns:
        Dict[str, Union[torch.Tensor, List[str]]]: 
            A dictionary containing the combined pixel values, padded input_ids, and ids.
    """

    batch = {}
    batch["flattened_patches"] = torch.stack([x["flattened_patches"][0] for x in samples])
    batch["attention_masks"] = torch.stack([x["attention_masks"][0] for x in samples])

    max_length = max([len(x["input_ids"]) for x in samples])

    # Make a multiple of 8 to efficiently use the tensor cores
    if max_length % 8 != 0:
        max_length = (max_length // 8 + 1) * 8

    input_ids = [
        x["input_ids"] + [pad_token_id] * (max_length - len(x["input_ids"]))
        for x in samples
    ]

    labels = torch.tensor(input_ids)
    labels[labels == pad_token_id] = -100 # ignore loss on padding tokens
    batch["labels"] = labels
    
    batch["id"] = [x["id"] for x in samples]

    return batch

if CFG.debug:
    train_ds = train_ds.select(range(100))
    val_ds = val_ds.select(range(10))

train_dataloader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
)
val_dataloader = DataLoader(
    val_ds,
    batch_size=CFG.val_batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
)


### Lightning Module
class DonutModelPLModule(pl.LightningModule):
    def __init__(self, processor: PreTrainedTokenizerBase, model: PreTrainedModel, gt_df: pd.DataFrame, num_training_steps: int):
        """
        A PyTorch Lightning module for the DonutModel.

        Args:
            processor (PreTrainedTokenizerBase): The tokenizer/processor for the model.
            model (PreTrainedModel): The pretrained model.
            gt_df (pd.DataFrame): The ground truth dataframe.
            num_training_steps (int): The number of training steps.
        """
        super().__init__()
        self.processor = processor
        self.model = model
        self.gt_df = gt_df
        self.num_training_steps = num_training_steps
        self.output_dir = ""
        self.mylog = {
            'val_score': [], 
            'horizontal_bar_score': [], 
            'line_score': [], 
            'scatter_score': [], 
            'vertical_bar_score': []
        }
        self.best_val_score = -1
    
    def register_output_dir(self, output_dir):
        self.output_dir = output_dir

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        flattened_patches = batch["flattened_patches"]
        attention_masks = batch["attention_masks"]
        labels = batch["labels"]

        outputs = self.model(flattened_patches=flattened_patches, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataset_idx: int = 0) -> None:
        flattened_patches = batch["flattened_patches"]
        attention_masks = batch["attention_masks"]

        outputs = self.model.generate(
            flattened_patches=flattened_patches,
            attention_mask=attention_masks,
            max_length=CFG.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            top_k=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        self.val_outputs.extend(
            self.processor.tokenizer.batch_decode(outputs.sequences)
        )
        self.val_ids.extend(batch["id"])
    
    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache() 

    def on_validation_start(self) -> None:
        self.val_outputs, self.val_ids = [], []

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        metrics = validation_metrics(self.val_outputs, self.val_ids, self.gt_df, save_path=os.path.join(self.output_dir, f"temp_{self.current_epoch}.csv"))
        print("\n", metrics)

        self.log_dict(metrics)

        self.val_outputs, self.val_ids = [], []
        
        ctypes = ["val_score", "horizontal_bar_score", "line_score", "scatter_score", "vertical_bar_score"]
        for ctype in ctypes:
            if ctype not in metrics:
                metrics[ctype] = 0.0
        self.mylog["val_score"].append(metrics["val_score"])
        self.mylog["horizontal_bar_score"].append(metrics["horizontal_bar_score"])
        self.mylog["line_score"].append(metrics["line_score"])
        self.mylog["scatter_score"].append(metrics["scatter_score"])
        self.mylog["vertical_bar_score"].append(metrics["vertical_bar_score"])
        mylog_df = pd.DataFrame.from_dict(self.mylog)
        mylog_df.to_csv(os.path.join(self.output_dir, "log.csv"), index=False)

        if self.best_val_score < metrics["val_score"]:
            print(f"Update! {self.best_val_score} -> {metrics['val_score']}")
            self.best_val_score = metrics["val_score"]
            self.model.save_pretrained(self.output_dir)
            self.processor.save_pretrained(self.output_dir)
        
        epoch_output_dir = os.path.join(self.output_dir, f"log/epoch_{self.current_epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        self.model.save_pretrained(epoch_output_dir)
        self.processor.save_pretrained(epoch_output_dir)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=CFG.lr, weight_decay=1e-05)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=int(len(train_df)//CFG.batch_size) * CFG.epochs)
        optimizers = [optimizer,]
        schedulers = [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        },]
        return optimizers, schedulers


### Model
num_training_steps = len(train_dataloader) * CFG.epochs // CFG.gpus
gt_chart_type = val_gt_ds["chart-type"]
gt_x = [json.loads(_) for _ in val_gt_ds["x"]]
gt_y = [json.loads(_) for _ in val_gt_ds["y"]]
gt_ids = val_gt_ds["id"]

index = [f"{id_}_x" for id_ in gt_ids] + [f"{id_}_y" for id_ in gt_ids]
gt_df = pd.DataFrame(
    index=index,
    data={
        "data_series": gt_x + gt_y,
        "chart_type": gt_chart_type * 2,
    },
)

print(f"model load from :{CFG.pretrain_weight}")
model = Pix2StructForConditionalGeneration.from_pretrained(CFG.pretrain_weight)
model.decoder.resize_token_embeddings(len(processor.tokenizer))
model_module = DonutModelPLModule(processor, model, gt_df, num_training_steps)

### Train
import datetime
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
d = now.strftime('%Y%m%d%H%M%S')
output_path = os.path.join(CFG.output_path, str(d))
os.makedirs(output_path, exist_ok=True)
model_module.register_output_dir(output_path)

checkpoint_cb = ModelCheckpoint(
    output_path,
    monitor="val_score",
    mode="max",
    save_top_k=1,
    save_last=False
)

loggers = []
if CFG.use_wandb:
    import wandb
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    key = user_secrets.get_secret("wandb")
    wandb.login(key=key)
    
    loggers.append(WandbLogger(project="benetech"))

trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=CFG.epochs,
        val_check_interval=CFG.val_check_interval,
        check_val_every_n_epoch=CFG.check_val_every_n_epoch,
        gradient_clip_val=CFG.gradient_clip_val,
        precision="bf16",
        num_sanity_val_steps=0,
        callbacks=[checkpoint_cb], 
        logger=loggers
)
trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)