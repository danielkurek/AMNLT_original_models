import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from jiwer import cer, wer
from peft import LoraConfig, get_peft_model
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

DEVICE = "cuda"


def load_dataset(dataset_name):
    # train: data-00000-of-00029.arrow  data-00005-of-00029.arrow  data-00010-of-00029.arrow  data-00015-of-00029.arrow  data-00020-of-00029.arrow  data-00025-of-00029.arrow  state.json
    # data-00001-of-00029.arrow  data-00006-of-00029.arrow  data-00011-of-00029.arrow  data-00016-of-00029.arrow  data-00021-of-00029.arrow  data-00026-of-00029.arrow
    # data-00002-of-00029.arrow  data-00007-of-00029.arrow  data-00012-of-00029.arrow  data-00017-of-00029.arrow  data-00022-of-00029.arrow  data-00027-of-00029.arrow
    # data-00003-of-00029.arrow  data-00008-of-00029.arrow  data-00013-of-00029.arrow  data-00018-of-00029.arrow  data-00023-of-00029.arrow  data-00028-of-00029.arrow
    # data-00004-of-00029.arrow  data-00009-of-00029.arrow  data-00014-of-00029.arrow  data-00019-of-00029.arrow  data-00024-of-00029.arrow
    train_ds = concatenate_datasets(
        [
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00000-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00001-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00002-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00003-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00004-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00005-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00006-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00007-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00008-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00009-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00010-of-00029.arrow"
            ),
            Dataset.from_file(
                "data/GregoSynth_hf_dataset/train/data-00011-of-00029.arrow"
            ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00012-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00013-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00014-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00015-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00016-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00017-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00018-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00019-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00020-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00021-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00022-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00023-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00024-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00025-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00026-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00027-of-00029.arrow"
            # ),
            # Dataset.from_file(
            #     "data/GregoSynth_hf_dataset/train/data-00028-of-00029.arrow"
            # ),
        ]
    )
    val_ds = Dataset.from_file("data/Solesmes_hf_dataset/val/data-00000-of-00001.arrow")
    test_ds = Dataset.from_file(
        "data/Solesmes_hf_dataset/test/data-00000-of-00001.arrow"
    )

    val_ds = val_ds.select(range(10))

    return train_ds, val_ds, test_ds


def load_model(model_name):
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )  # , quantization_config=bnb_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


@record
def main(args):
    import gc

    torch.cuda.empty_cache()
    gc.collect()

    train_ds, val_ds, test_ds = load_dataset(args.dataset)

    model = load_model(args.model)
    processor = PaliGemmaProcessor.from_pretrained(args.model)

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    DTYPE = model.dtype

    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

    def collate_fn(examples):
        texts = ["<image>answer en " + example["question"] for example in examples]
        labels = [example["multiple_choice_answer"] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
        )

        tokens = tokens.to(DTYPE).to(DEVICE)
        # RuntimeError: aten.cat.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!

        return tokens

    args = TrainingArguments(
        num_train_epochs=50,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=50,
        optim="paged_adamw_8bit",  # you can use paged optimizers like paged_adamw_8bit for QLoRA
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        output_dir="weights/paligemma_vqa-3b-896/GregoSynth",
        bf16=True,
        report_to=["wandb"],
        run_name="paligemma2-3b-896-GregoSynth",
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )

    # args = TrainingArguments(
    #     num_train_epochs=100,
    #     remove_unused_columns=False,
    #     per_device_train_batch_size=1,
    #     gradient_accumulation_steps=1,
    #     warmup_steps=2,
    #     learning_rate=2e-5,
    #     weight_decay=1e-6,
    #     adam_beta2=0.999,
    #     logging_steps=10,
    #     optim="paged_adamw_8bit",
    #     save_strategy="steps",
    #     save_steps=1000,
    #     save_total_limit=1,
    #     output_dir="weights/paligemma_vqa-3b-pt-896/GregoSynth",
    #     bf16=True,
    #     report_to=["wandb"],
    #     # report_to=["wandb"] if is_main_process else [],
    #     dataloader_pin_memory=False,
    #     # do_eval=True,
    #     # per_device_eval_batch_size=1,
    #     gradient_checkpointing=True,
    #     # eval_strategy="steps",
    #     # eval_steps=args.eval_steps,
    #     # eval_accumulation_steps=1,
    #     logging_strategy="steps",
    #     ddp_find_unused_parameters=False,
    # )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        data_collator=collate_fn,
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/paligemma2-3b-pt-896",
        help="Model name",
        choices=["google/paligemma2-3b-pt-896", "google/paligemma2-3b-pt-224"],
    )
    parser.add_argument(
        "--dataset", type=str, default="GregoSynth_hf_dataset", help="Dataset name"
    )
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument(
        "--ds_config",
        type=str,
        default="ds_config.json",
        help="Path to deepspeed config",
    )
    args = parser.parse_args()
    main(args)
    # torchrun --nproc_per_node 2  src/pr_amnlt_foundation/simple_vlm.py