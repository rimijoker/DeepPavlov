# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from typing import List, Union, Iterable

import numpy as np

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component


@register('multitask_pal_bert_preprocessor')
class MultitaskPalBertPreprocessor(Component):
    """
    One-hot featurizer with zero-padding.
    If ``single_vector``, return the only vector per sample which can have several elements equal to ``1``.

    Parameters:
        depth: the depth for one-hotting
        pad_zeros: whether to pad elements of batch with zeros
        single_vector: whether to return one vector for the sample (sum of each one-hotted vectors)
    """

    def __init__(self, *args, **kwargs):
        print(args, kwargs)

    def __call__(self, *args):
        # print("$$ARGS$$",args)
        out = []
        task_id = None
        for examples in zip(*args):
            example_with_task_id = []
            for task_no, task_example in enumerate(examples):
                current_task_id = task_example[0]
                if task_id is not None:
                    if current_task_id != task_id:
                        raise ValueError(f"Two task_ids found {current_task_id}, {task_id}"
                        "One batch should not have multiple task_ids")
                if task_no == 0:
                    # index zero will be task_id
                    task_id = current_task_id
                    example_with_task_id = [task_id, *task_example[1:]] 
                else:
                    example_with_task_id.extend([*task_example[1:]])
            out.append(tuple(example_with_task_id))
        print("MultitaskPalBertPreprocessor", tuple(out))

        return tuple(out)