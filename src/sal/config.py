#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Set


@dataclass
class ModelConfig:
    base_path: Optional[Path] = None  # when set to empty string it assumes hfhub
    name: str = ""

    def get_model_path(self) -> str:
        if self.base_path is not None:
            return f"{self.base_path}/{self.name}"
        return self.name


@dataclass
class GeneratorConfig(ModelConfig):
    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    parameter_count: Optional[str] = None
    quantisation: Optional[str] = None
    max_model_len: int = 6144
    gpu_memory_utilization: float = 0.5


@dataclass
class PRMConfig(ModelConfig):
    name: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"


@dataclass
class ProfilerConfig:
    # memory profiling
    profile_memory: bool = True
    memory_snapshot_file: str = "memory_snapshot.pickle"
    memory_max_entries: int = 10000
    # operation profiling
    profile_operations: bool = True
    operations_trace_dir: str = "trace"


@dataclass
class DatasetConfig:
    dataset_name: Literal["HuggingFaceH4/aime_2024", "HuggingFaceH4/MATH-500"] = (
        "HuggingFaceH4/MATH-500"
    )
    dataset_config: Optional[str] = None
    dataset_split: str = "test"
    dataset_start: Optional[int] = None
    dataset_end: Optional[int] = None
    num_samples: Optional[int] = None


@dataclass
class OutputConfig:
    output_dir_base: str = "output"
    num_proc: Optional[int] = None
    apply_voting: bool = True
    # contains the generated output, PRM scores and voting results
    inference_output_file: str = "inference_output.jsonl"
    # contans score on evaluation datasets
    evaluation_score_file: str = "evaluation_score.jsonl"


@dataclass
class SearchConfig:
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 25
    max_tokens: int = 2048
    agg_strategy: Literal["last", "min", "prod"] = "last"


@dataclass
class BeamSearchConfig:
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40
    lookahead: int = 0


@dataclass
class QCConfig:
    high_threshold: float = 0.8
    low_threshold: float = 0.3
    # Optional dynamic CMC knobs (used by q2). Defaults keep existing behavior if unused.
    use_dynamic_thresholds: bool = True
    high_q: float = 0.8
    low_q: float = 0.2
    delta_high: float = 0.10
    sigma_low: float = 0.10
    min_beams_for_quantiles: int = 8
    # Single-knob compute control. If set, overrides quantiles to target upgrade rate
    # and disables margin/stability gating for simplicity.
    target_upgrade_rate: float | None = None  # in [0, 1]
    enable_margin_stability: bool = True


@dataclass
class WandbConfig:
    project: str = "qtts"
    tags: Set[str] = field(default_factory=lambda: set())


@dataclass
class EvaluationConfig:
    benchmark: str = "math"
    dataset_col: str = "pred"


@dataclass
class BaseConfig:
    """Configuration that remains constant across all experiments"""

    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    profiler_config: ProfilerConfig = field(default_factory=ProfilerConfig)

    # potentially might want to make this expriment config in the future?
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)
    draft_config: Optional[GeneratorConfig] = None
    prm_config: PRMConfig = field(default_factory=PRMConfig)
    seed: int = 0


@dataclass
class ExperimentConfig:
    """Configuration that varies between experiments"""

    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    search_config: SearchConfig = field(default_factory=SearchConfig)
    beam_search_config: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    qcconfig: QCConfig = field(default_factory=QCConfig)

    approach: Literal[
        "best_of_n", "beam_search", "dvts", "qcts", "q2", "diagnostic_tts"
    ] = "best_of_n"

    # Chat template related options
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    custom_chat_template: Optional[str] = (
        '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    )

    filter_duplicates: bool = False
    sort_completed: bool = False
    seed: int = 0

    def __post_init__(self):
        if self.approach in ["dvts", "qcts", "q2", "diagnostic_tts"]:
            if self.search_config.n % self.beam_search_config.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.search_config.n // self.beam_search_config.beam_width

        if self.approach == "beam_search":
            # TODO: implemented a batched version
            if self.search_config.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for beam_search")
