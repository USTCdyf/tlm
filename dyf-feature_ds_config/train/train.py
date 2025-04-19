# train.py
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import torch
from models.TLM import TimeLLMConfig, TimeLLMModel, MeanScaleQuantileBins
from config import get_time_llm_config


# 2. 数据集处理（使用load_dataset）
def prepare_dataset(file_path: str, config):
    # 加载JSON数据（自动支持缓存）
    dataset = load_dataset("json", data_files=file_path, split="train")

    # 初始化tokenizers
    bin_tokenizer = config.create_tokenizer()
    llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_name)

    def process_example(example):
        # 时序数据分箱
        ts_values = [
            float(x) for x in example["instruction"].split(":")[1].strip().split(",")
        ]
        bin_ids, _, scale = bin_tokenizer.context_input_transform(
            torch.tensor([ts_values])
        )

        # 构造输入输出文本
        prompt = example["input"]
        answer = example["output"]

        # Qwen 使用的聊天格式
        # TODO SYSTEM暂时自己写的的，建议改为和自己任务更契合的prompt
        messages = [
            {
                "role": "system",
                "content": "Based on the historical one-day electricity load time series, as well as the future weather and time information, predict the electricity load changes in the specified region for the next day. Please output exactly 96 data points only, and nothing else.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

        full_text = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize（自动处理截断和填充）
        tokenized = llm_tokenizer(
            full_text, truncation=False, padding=False, return_tensors=None
        )

        # 计算loss mask
        prompt_only_messages = messages[:-1]  # 去掉 assistant 回复
        prompt_text = llm_tokenizer.apply_chat_template(
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )
        input_len = len(
            llm_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        labels = [-100] * config.query_len + tokenized[
            "input_ids"
        ].copy()  # 前16是时序部分
        labels[config.query_len : config.query_len + input_len] = [
            -100
        ] * input_len  # 标记instruction

        return {
            "bin_ids": bin_ids[0].tolist(),
            "input_ids": tokenized["input_ids"],
            "attention_mask": [1]
            * (config.query_len + len(tokenized["input_ids"])),  # 时序+文本
            "labels": labels,
            "scale": scale.item(),
        }

    # 应用预处理（启用缓存）
    processed_dataset = dataset.map(
        process_example, batched=False, remove_columns=dataset.column_names
    )

    return processed_dataset


def get_collate_fn(llm_tokenizer, config):
    """闭包方式传递tokenizer和config"""

    def collate_fn(batch):
        max_text_len = max(len(x["input_ids"]) for x in batch)
        total_len = config.query_len + max_text_len

        def pad_field(values, pad_value, leng):
            return torch.stack(
                [
                    torch.cat(
                        [
                            torch.tensor(v, dtype=torch.long),
                            torch.full((leng - len(v),), pad_value),
                        ]
                    )
                    for v in values
                ]
            )

        return {
            "bin_ids": torch.stack([torch.tensor(x["bin_ids"]) for x in batch]),
            "input_ids": pad_field(
                [x["input_ids"] for x in batch],
                llm_tokenizer.pad_token_id,
                max_text_len,  # 使用传入的tokenizer
            ),
            "attention_mask": pad_field(
                [x["attention_mask"] for x in batch], 0, total_len
            ),
            "labels": pad_field([x["labels"] for x in batch], -100, total_len),
            "scales": torch.tensor([x["scale"] for x in batch]),
        }

    return collate_fn


# 4. 训练主函数
def train(config):
    # 加载处理后的数据
    train_dataset = prepare_dataset(
        "/home/scb123/PyProject/tlm/dyf-feature_ds_config/data/AULF_train_data_2019-2020.json",
        config,
    )
    val_dataset = prepare_dataset(
        "/home/scb123/PyProject/tlm/dyf-feature_ds_config/data/AULF_test_data_2021.json",
        config,
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_name)
    collate_fn = get_collate_fn(
        llm_tokenizer=llm_tokenizer, config=config
    )  # 传入tokenizer

    # 初始化模型
    model = TimeLLMModel(config)
    # model = model.to(torch.bfloat16)

    # 训练参数（添加wandb支持）
    training_args = TrainingArguments(
        output_dir="./time_llm_results",
        weight_decay=0.01,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=10,
        fp16=True,
        deepspeed="/opt/tiger/dyf/train/ds_config.json",
        remove_unused_columns=False,
        report_to=["wandb"],  # 可选
        save_total_limit=2,
        eval_strategy="steps",  # 添加evaluation策略
        eval_steps=10,  # 每100步进行一次验证
        load_best_model_at_end=True,  # 在训练结束时加载并保存最好的模型
        metric_for_best_model="eval_loss",  # 使用验证集上的损失作为评估指标
        greater_is_better=False,  # 损失越低越好
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 启动训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model("./final_model")
    print(f"训练完成！模型已保存到: ./final_model")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = get_time_llm_config()

    train(config=config)
