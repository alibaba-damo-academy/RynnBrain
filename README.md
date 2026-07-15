# RynnBrain
<p align="center">
<img src="./cookbooks/assets/logo.png" style="width: 50%; height: auto;">
</p>
<p align="center">
       💫 <a href="https://alibaba-damo-academy.github.io/RynnBrain.github.io/"><b>Project Page</b></a>&nbsp;&nbsp; | &nbsp;&nbsp; 🤗 <a href ="https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain"><b> Hugging Face </b></a> &nbsp;&nbsp; | &nbsp;&nbsp; 🤖 <a href = "https://www.modelscope.cn/collections/DAMO_Academy/RynnBrain"><b> ModelScope</b></a>  &nbsp;|&nbsp; 🚀 <a href="https://huggingface.co/spaces/Alibaba-DAMO-Academy/RynnBrain"><b>Demo</b></a> &nbsp;&nbsp; | &nbsp;&nbsp;📚 <a href="https://github.com/alibaba-damo-academy/RynnBrain/tree/main/cookbooks">Cookbooks</a>&nbsp;&nbsp; | &nbsp;&nbsp; 📄 <a href="https://arxiv.org/abs/2602.14979v1">arXiv</a>&nbsp;&nbsp;

</p>


## 📰 News
* **[2026.04.13]**  🔥🔥 Launch the brand-new <a href="https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-4B">RynnBrain-4B </a> !!
* **[2026.02.17]**  🔥🔥 Release our Technical Report on <a href="https://arxiv.org/abs/2602.14979v1">arXiv </a> !!
* **[2026.02.15]**  🔥🔥 Release our <a href="https://alibaba-damo-academy.github.io/RynnBrain.github.io/assets/RynnBrain_Report.pdf">Technical Report</a> !!
* **[2026.02.09]**  🔥🔥 Release our code and model checkpoints!!



## Introduction
We present **RynnBrain**, an embodied foundation model grounded in physical reality. RynnBrain is available in three dense variants (2B, 4B, and 8B) and one mixture-of-experts (MoE) model (30B-A3B). 
In addition, we release three post‑trained models: RynnBrain‑Plan (**robot task planning**), RynnBrain‑Nav (**vision-language navigation**), and RynnBrain‑CoP (**chain-of-point reasoning**). 
<!-- RynnBrain‑Plan demonstrates the effectiveness of the fine‑grained manipulation‑planning paradigm that alternates between textual reasoning and localization. -->
<!-- RynnBrain‑Nav verifies that using RynnBrain as the foundation model can substantially enhance the performance ceiling of various embodied task models.  -->
<!-- Brain-CoP incorporates an interleaved reasoning mechanism that alternates between textual reasoning and spatial grounding, endowing it with physical-space reasoning capabilities.  -->

### 🌟 Key Highlights
* **Comprehensive egocentric understanding**: 
Excels in fine-grained video understanding and egocentric cognition, covering tasks such as embodied QA, counting, and OCR.
* **Diverse spatio-temporal localization**: 
Possesses powerful localization capabilities across episodic memory, enabling precise identification of objects, target areas, and motion trajectories.
* **Physical-space reasoning**: 
Employs an interleaved reasoning strategy that alternates between textual and spatial grounding, ensuring that its reasoning processes are firmly rooted in the physical environment.
* **Physics-aware precise planning**: 
Integrates located affordances and object information into planning, enabling downstream VLA models to execute intricate tasks with fine-grained instructions.

<p align="center">
<img src="./cookbooks/assets/intro.png" style="width: 90%; height: auto;">
</p>

### Model Architecture
RynnBrain employs a unified architecture (supporting both Dense and MoE variants) to transform omni-vision inputs and textual instructions into multi-modal outputs, including spatial trajectories, physical pointing, and action planning. 
Through massive training on rich spatio-temporal, physical-space, and general knowledge data, RynnBrain maintains robust general-purpose capabilities while specializing in diverse, fine-grained embodied reasoning and complex planning tasks.

<p align="center">
<img src="./cookbooks/assets/framework.png" style="width: 90%; height: auto;">
</p>

## Performance

- General Embodied Understanding

<p align="center">
<img src="./cookbooks/assets/performance_general_2B_8B.png" style="width: 80%; height: auto;">
</p>
<p align="center">
<img src="./cookbooks/assets/performance_general_30B.png" style="width: 80%; height: auto;">
</p>


- Robot Task Planning

<p align="center">
<img src="https://github.com/user-attachments/assets/ce0b20c2-81be-403c-bd5f-19bbe5235dd2" style="width: 80%; height: auto;">
</p>


- Vision-Language Navigation

<p align="center">
<img src="https://github.com/user-attachments/assets/78c36b4e-0ea8-42e2-a3fd-692d7c2fb4a7" style="width: 80%; height: auto;">
</p>


## Model Zoo

| Model            | Base Model           | HuggingFace | ModelScope |
| :--------------- | :------------------- | :---------: | :--------: |
| RynnBrain-2B  | Qwen3-VL-2B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-2B)   |
| RynnBrain-4B  | Qwen3-VL-4B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-4B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-4B)   |
| RynnBrain-8B  | Qwen3-VL-8B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-8B)   |
| RynnBrain-30B-A3B  | Qwen3-VL-30B-A3B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-30B-A3B)   |
| RynnBrain‑CoP-8B | RynnBrain-8B         | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-CoP-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-CoP-8B)   |
| RynnBrain‑Plan-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-8B)   |
| RynnBrain‑Plan-30B-A3B | RynnBrain-30B-A3B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-30B-A3B)   |
| RynnBrain‑Nav-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Nav-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Nav-8B)   |



## Quick Start

### Inference with 🤗transformers

**Minimal dependencies**
```shell
pip install transformers==4.57.1
```
**Run text generation**
```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

conversation = [
    {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': 'cookbooks/assets/object_location/images/000000086408.jpg'},
            {'type': 'text', 'text': 'What appliance can be used to heat food quickly.\nGenerate coordinates for one object bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>'},
        ],
    }
]

model_path = "Alibaba-DAMO-Academy/RynnBrain-2B"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
)
model.to("cuda")

model_inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = model_inputs.to("cuda")

output_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
)
output_ids = output_ids[:, model_inputs["input_ids"].size(1):]
response = processor.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

> **Note**: RynnBrain also runs on Intel XPU (integrated and discrete GPUs) via PyTorch's native XPU backend — simply replace `"cuda"` with `"xpu"` in the two `.to(...)` calls above. No other code changes are required. Verified on RynnBrain-4B (Qwen3-VL-4B-Instruct base): CPU vs XPU logits MSE = 2.9e-07, greedy-decoded output identical.



### Inference with SGLang

For installation and advanced usages, please refer to the official [documentation](https://docs.sglang.io).

**OpenAI-Compatible Serving**
```shell
# launch server
python3 -m sglang.launch_server --model-path Alibaba-DAMO-Academy/RynnBrain-2B --host 0.0.0.0 --port 8000
```

```python
# inference using openai api
import base64
import io

from openai import OpenAI
from PIL import Image

def pil_to_url(image: Image.Image):
    image_format = image.format if image.format else 'PNG'
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/{image_format.lower()};base64,{img_str}'

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': pil_to_url(Image.open('cookbooks/assets/object_location/images/000000086408.jpg'))}},
            {'type': 'text', 'text': 'What appliance can be used to heat food quickly.\nGenerate coordinates for one object bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>'},
        ],
    }
]

client = OpenAI(api_key="", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="default",
    messages=messages,
    stream=False,
).choices[0].message.content
print(response)
```

**Offline Engine**
```python
import sglang as sgl
from transformers import AutoProcessor

def main():
    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'What appliance can be used to heat food quickly.\nGenerate coordinates for one object bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>'},
            ],
        }
    ]

    model_path = 'Alibaba-DAMO-Academy/RynnBrain-2B'
    llm = sgl.Engine(model_path=model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    output = llm.generate(
        prompt=prompt,
        image_data='cookbooks/assets/object_location/images/000000086408.jpg',
        sampling_params={"temperature": 0.8, "top_p": 0.95},
    )
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == '__main__':
    main()
```



## Cookbooks
Checkout the [cookbooks](./cookbooks) that showcase RynnBrain's capabilities in cognition, localization, reasoning, and planning.


| Category             | Cookbook name                                                                                   | Description |
|----------------------|--------------------------------------------------------------------------------------------------|-------------|
| Cognition            | [1_spatial_understanding.ipynb](./cookbooks/1_spatial_understanding.ipynb)                     | Shows the model's ability for spatial understanding in the video scene. |
| Cognition            | [2_object_understanding.ipynb](./cookbooks/2_object_understanding.ipynb)                       | Shows how the model understands object categories, attributes, and relations and counting ability. |
| Cognition            | [3_ocr.ipynb](./cookbooks/3_ocr.ipynb)                                                         | Examples of optical character recognition and text understanding in videos. |
| Location             | [4_object_location.ipynb](./cookbooks/4_object_location.ipynb)                                 | Locates specific objects with bounding boxes in an image or video based on instructions. |
| Location             | [5_area_location.ipynb](./cookbooks/5_area_location.ipynb)                                     | Identifies and marks specified regions by points in an image or video. |
| Location             | [6_affordance_location.ipynb](./cookbooks/6_affordance_location.ipynb)                         | Finds areas or objects with specific affordances in an image or video. |
| Location             | [7_trajectory_location.ipynb](./cookbooks/7_trajectory_location.ipynb)                         | Infers and annotates trajectories or motion paths in an image or video. |
| Location             | [8_grasp_pose.ipynb](./cookbooks/8_grasp_pose.ipynb)                                           | Presents the model's ability to predict robotic grasp poses from images. |
| Reasoning            | [9_thinking_with_time_space.ipynb](./cookbooks/9_thinking_with_time_space.ipynb)               | Explores an interleaved reasoning mechanism that alternates between textual reasoning and spatial grounding. |
| Planning | [10_manipulate_planning.ipynb](./cookbooks/10_manipulate_planning.ipynb)                                     | Performs multi-step task decomposition and action planning from goals and scenes. |
| Planning | [11_visual_language_navigation.ipynb](./cookbooks/11_visual_language_navigation.ipynb)         | Combines vision and language instructions to perform navigation and path planning. |


## Training

**Pretraining & Evaluation** 

Please refer to [RynnScale](https://github.com/alibaba-damo-academy/RynnScale/tree/main/projects/rynn_brain) for details of pretraining and evaluation.


**Finetuning**

- [Reasoning](reasoning): RynnBrain introduces an **interleaved reasoning approach that combines grounding with textual information** directly within egocentric video streams. This paradigm effectively bridges the cognitive gap between language and the physical world, ensuring the reasoning process is robustly anchored
in reality. 

- [Navigation](navigation):
We trained a vision-language navigation model based on the RynnBrain base model. Empirical evaluation demonstrates that fine-tuning the vision-language model on RynnBrain yields superior performance compared to fine-tuning on other foundational models.

- [Planning](planning):
RynnBrain **integrates the location information of affordance, areas, and objects directly
into its planning outputs**. Consequently, even highly intricate and fine-grained tasks can be effectively addressed within our hierarchical RynnBrain-VLA system architecture.


## RynnBrain-Bench
We introduce **RynnBrain-Bench**, a high-dimensional benchmark for embodied understanding that evaluates models across four key dimensions: *object cognition*, *spatial cognition*, *grounding*, and *pointing*—highlighting fine-grained understanding and spatiotemporal localization across episodic video sequences.

For details, please refer to [RynnBrain-Bench](./rynnbrain-bench/README.md).
<p align="center">
<img src="./cookbooks/assets/RynnBrain-Bench.png" style="width: 80%; height: auto;">
</p>


## 📑 Citation

If you find RynnBrain useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{damo2026rynnbrain,
  title={RynnBrain: Open Embodied Foundation Models},
  author={Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu, Xiao Lin, Yang Bai, Qian Jiang, Yaxi Zhao, Minghua Zeng, Junlong Gao, Yuming Jiang, Jun Cen, Siteng Huang, Liuyi Wang, Wenqiao Zhang, Chengju Liu, Jianfei Yang, Shijian Lu, Deli Zhao},
  journal={arXiv preprint arXiv:2602.14979v1},
  year={2026},
  url = {https://arxiv.org/abs/2602.14979v1}
}

```

<details open><summary>💡 Some other multimodal-LLM projects from our team may interest you ✨. </summary><p>
<!--  may -->
       
> [**RynnEC: Bringing MLLMs into Embodied World**](https://github.com/alibaba-damo-academy/RynnEC) <br>
> Ronghao Dang*, Yuqian Yuan*, Yunxuan Mao*, Kehan Li*, Jiangpin Liu, Zhikai Wang, Fan Wang, Deli Zhao, Xin Li <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnEC)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnEC.svg?style=social)](https://github.com/alibaba-damo-academy/RynnEC) [![arXiv](https://img.shields.io/badge/Arxiv-2508.14160-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.14160) <br>
       
> [**RynnScale**](https://github.com/alibaba-damo-academy/RynnScale) <br>
> RynnScale Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnScale)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnScale.svg?style=social)](https://github.com/alibaba-damo-academy/RynnScale) <br>

> [**RynnVLA-001: Using Human Demonstrations to Improve Robot Manipulation**](https://arxiv.org/abs/2509.15212) <br>
> Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, Fan Wang, Deli Zhao, Xin Li <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnVLA-001)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnVLA-001.svg?style=social)](https://github.com/alibaba-damo-academy/RynnVLA-001)  [![arXiv](https://img.shields.io/badge/Arxiv-2509.15212-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.15212) <br>

> [**RynnVLA-002: A Unified Vision-Language-Action and World Model**](https://arxiv.org/abs/2511.17502) <br>
> Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, Fan Wang, Deli Zhao, Hao Chen <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnVLA-002)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnVLA-002.svg?style=social)](https://github.com/alibaba-damo-academy/RynnVLA-002)  [![arXiv](https://img.shields.io/badge/Arxiv-2511.17502-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.17502) <br>

> [**RynnRCP: Open Robotics Context Protocol and RobotMotion**](https://github.com/alibaba-damo-academy/RynnRCP) <br>
> RynnBot Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnRCP)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnRCP.svg?style=social)](https://github.com/alibaba-damo-academy/RynnRCP)  <br>

> [**RynnMotion: All-In-One Toolkit for Fast Robot Prototyping and Heterogeneous Teleoperation**](https://github.com/alibaba-damo-academy/RynnMotion) <br>
> RynnBot Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnMotion)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnMotion.svg?style=social)](https://github.com/alibaba-damo-academy/RynnMotion)  <br>

</p></details>

## Acknowledgement

Our RynnBrain is built on top of [**Qwen3-VL**](https://github.com/QwenLM/Qwen3-VL). We also learned a lot from the implementation of [**RynnEC**](https://github.com/alibaba-damo-academy/RynnEC) and [**VideoRefer**](https://github.com/DAMO-NLP-SG/VideoRefer). If your work is used in RynnBrain but not mentioned in either this repo or the technical report, feel free to let us know :heart:.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
