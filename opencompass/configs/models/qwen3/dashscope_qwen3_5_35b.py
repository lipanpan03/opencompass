from opencompass.models import DashScopeAPI

# Meta template for chat-style models.
# Maps OpenCompass internal roles to API roles.
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        abbr='dashscope-qwen3.5-35b',
        type=DashScopeAPI,
        # The model name sent in the request body
        path='pre-qwen3.5-35b-a3b-1tp-fp4',
        # Full DashScope multimodal-generation endpoint
        url='https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
        # Replace with your actual Bearer token, or use env var via os.environ
        key='sk-f40d129e39134c1ba4329e581bd87f53',
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        # Extra parameters forwarded into the "parameters" field of the request
        generation_kwargs=dict(
            incremental_output=False,
        ),
        # Set to True if you want reasoning_content prepended to the answer
        with_reasoning=False,
    ),
]

