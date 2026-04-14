import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class DashScopeAPI(BaseAPIModel):
    """Model wrapper for DashScope multimodal-generation API.

    Supports the DashScope protocol used by Alibaba Cloud's model services.
    The API endpoint format is:
        POST /api/v1/services/aigc/multimodal-generation/generation

    Request body format::

        {
            "model": "<model_name>",
            "input": {
                "messages": [{"role": "user", "content": "..."}]
            },
            "parameters": {
                "max_length": 512,
                "incremental_output": false
            }
        }

    Response body format::

        {
            "output": {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "..."}],
                        "reasoning_content": "..."
                    }
                }]
            },
            "usage": {...},
            "request_id": "..."
        }

    Args:
        path (str): The model name to use, e.g. ``pre-qwen3.5-35b-a3b-1tp-fp4``.
        url (str): The full API endpoint URL.
        key (str): The Bearer token for Authorization header.
        query_per_second (int): Max queries per second. Defaults to 1.
        max_seq_len (int): Maximum sequence length. Defaults to 4096.
        meta_template (Dict, optional): Meta prompt template for role mapping.
        retry (int): Number of retries on failure. Defaults to 5.
        generation_kwargs (Dict): Extra parameters passed into ``parameters``
            field of the request body, e.g. ``{"incremental_output": false}``.
        with_reasoning (bool): Whether to prepend ``reasoning_content`` before
            the final answer text in the returned string. Defaults to False.
        max_workers (int, optional): Max thread workers for concurrent calls.
            Defaults to None (auto-detected from CPU count).
    """

    is_api: bool = True

    def __init__(
        self,
        path: str,
        url: str,
        key: str,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        max_seq_len: int = 4096,
        meta_template: Optional[Dict] = None,
        retry: int = 5,
        generation_kwargs: Optional[Dict] = None,
        with_reasoning: bool = False,
        max_workers: Optional[int] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs or {},
        )
        self.url = url
        self.key = key
        self.with_reasoning = with_reasoning

        if max_workers is None:
            import os
            cpu_count = os.cpu_count() or 1
            self.max_workers = min(32, cpu_count * 2)
        else:
            self.max_workers = max_workers

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if len(inputs) == 1:
            return [self._generate(inputs[0], max_out_len)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                    ),
                    total=len(inputs),
                    desc='Inferencing',
                ))
        self.flush()
        return results

    def _build_messages(self, input: PromptType) -> List[Dict]:
        """Convert OpenCompass PromptType to DashScope messages format.

        Args:
            input (PromptType): A string or PromptList from OpenCompass.

        Returns:
            List[Dict]: A list of message dicts with ``role`` and ``content``.
        """
        if isinstance(input, str):
            return [{'role': 'user', 'content': input}]

        messages = []
        msg_buffer, last_role = [], None

        for index, item in enumerate(input):
            if index == 0 and item['role'] == 'SYSTEM':
                role = 'system'
            elif item['role'] == 'BOT':
                role = 'assistant'
            else:
                role = 'user'

            if role != last_role and last_role is not None:
                messages.append({
                    'role': last_role,
                    'content': '\n'.join(msg_buffer),
                })
                msg_buffer = []

            msg_buffer.append(item['prompt'])
            last_role = role

        if msg_buffer:
            messages.append({
                'role': last_role,
                'content': '\n'.join(msg_buffer),
            })

        return messages

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate a single result for one input.

        Args:
            input (PromptType): A string or PromptList.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        messages = self._build_messages(input)

        parameters = {'max_length': max_out_len, 'incremental_output': False}
        parameters.update(self.generation_kwargs)

        request_body = {
            'model': self.path,
            'input': {'messages': messages},
            'parameters': parameters,
        }

        headers = {
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json',
        }

        num_retries = 0
        response = None

        while num_retries < self.retry:
            self.acquire()
            try:
                raw_response = requests.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(request_body),
                    timeout=120,
                )
            except requests.ConnectionError as connection_error:
                self.logger.error(f'Connection error: {connection_error}')
                self.release()
                time.sleep(2)
                num_retries += 1
                continue
            except requests.Timeout:
                self.logger.error('Request timed out, retrying...')
                self.release()
                time.sleep(2)
                num_retries += 1
                continue

            self.release()

            if raw_response.status_code == 429:
                self.logger.warning('Rate limit exceeded, waiting 5s...')
                time.sleep(5)
                num_retries += 1
                continue

            if raw_response.status_code != 200:
                self.logger.error(
                    f'Request failed with status {raw_response.status_code}: '
                    f'{raw_response.text}')
                num_retries += 1
                continue

            try:
                response = raw_response.json()
            except requests.JSONDecodeError as decode_error:
                self.logger.error(f'JSON decode error: {decode_error}')
                num_retries += 1
                continue

            try:
                choice = response['output']['choices'][0]
                message = choice['message']

                # content is a list of dicts, e.g. [{"text": "..."}]
                content_list = message.get('content', [])
                if isinstance(content_list, list):
                    answer_text = ''.join(
                        item.get('text', '') for item in content_list
                        if isinstance(item, dict)
                    )
                else:
                    # Fallback: content may be a plain string in some variants
                    answer_text = str(content_list)

                reasoning_content = message.get('reasoning_content', '') or ''

                self.logger.debug(f'Answer: {answer_text}')

                if self.with_reasoning and reasoning_content:
                    return reasoning_content + '</think>' + answer_text

                return answer_text.strip()

            except (KeyError, IndexError, TypeError) as parse_error:
                self.logger.error(
                    f'Failed to parse response: {parse_error}, '
                    f'response body: {response}')
                num_retries += 1
                continue

        raise RuntimeError(
            f'DashScope API call failed after {num_retries} retries. '
            f'Last response: {response}')

