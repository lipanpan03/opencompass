from mmengine.config import read_base

with read_base():
    # 1. MMLU-Pro: 多选题知识评测（10选项，0-shot COT）
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets

    # 2. MMLU-Redux (MMLU-CF): 去污染版 MMLU，4选项，5-shot
    from opencompass.configs.datasets.mmlu_cf.mmlu_cf_gen import mmlu_cf_datasets

    # 3. C-Eval: 中文知识评测
    # 需要先下载数据集，执行：
    #   export HF_ENDPOINT=https://hf-mirror.com
    #   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='opencompass/ceval-exam', repo_type='dataset', local_dir='./data/ceval/formal_ceval')"
    # from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets

    # 4. SuperGPQA: 超难研究生级别知识评测
    from opencompass.configs.datasets.supergpqa.supergpqa_gen import supergpqa_datasets

    # 模型配置：DashScope Qwen3.5-35B
    from opencompass.configs.models.qwen3.dashscope_qwen3_5_35b import models

# 每个子数据集只取前 50 条样本，快速验证
for dataset in mmlu_pro_datasets + mmlu_cf_datasets + supergpqa_datasets:
    dataset['reader_cfg'] = dataset.get('reader_cfg', {})
    dataset['reader_cfg']['test_range'] = '[0:50]'

# 合并所有数据集（C-Eval 待下载数据后取消注释并加入）
datasets = (
    mmlu_pro_datasets
    + mmlu_cf_datasets
    # + ceval_datasets
    + supergpqa_datasets
)
