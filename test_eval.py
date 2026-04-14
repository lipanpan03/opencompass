from mmengine.config import read_base

with read_base():
    # 使用一个小数据集测试
    from .datasets.demo.demo_gen import demo_datasets

models = [
    dict(
        # 如果是纯文本评测，先试这个类
        type='opencompass.models.DashScope', 
        path='pre-qwen3.5-35b-a3b-1tp-fp4', # 模型名称
        key='sk-f40d129e39134c1ba4329e581bd87f53', # API Key
        is_chat=True,
        # 这里的参数会传递给 dashscope.Generation.call
        generation_kwargs=dict(
            # 如果你的接口强制要求 multimodal，可能需要指定参数
            # 但通常 DashScope SDK 会根据模型名自动路由
        ),
        batch_size=1, # 建议先从 1 开始测试
    )
]

datasets = demo_datasets

