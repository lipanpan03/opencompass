from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.models.qwen3.dashscope_qwen3_5_35b import models

# 只取前 50 条样本，快速验证
for dataset in gsm8k_datasets:
    dataset['abbr'] = dataset.get('abbr', 'gsm8k') + '_50'
    dataset['reader_cfg'] = dict(
        input_columns=['question'],
        output_column='answer',
        test_range='[0:50]',   # 只取前 50 条
    )

datasets = gsm8k_datasets
