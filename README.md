# RynnBrain
<p align="center">
<img src="./cookbooks/assets/logo.png" style="width: 50%; height: auto;">
</p>
<p align="center">
       💫 <a href="https://alibaba-damo-academy.github.io/RynnBrain.github.io/"><b>Project Page</b></a>&nbsp&nbsp | &nbsp&nbsp Models <a href ="https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain"><b> 🤗</b></a> <a href = "https://www.modelscope.cn/collections/DAMO_Academy/RynnBrain"><b>🤖</b></a>  &nbsp;|&nbsp; 🚀 <a href="https://huggingface.co/spaces/Alibaba-DAMO-Academy/RynnBrain"><b>Demo</b></a> &nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/alibaba-damo-academy/RynnBrain/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp
</p>

<details open><summary>💡 Some other multimodal-LLM projects from our team may interest you ✨. </summary><p>
<!--  may -->
       
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

> [**RynnMotion: All-In-One Tookit for Fast Robot Prototyping and Heterogeneous Teleoperation**](https://github.com/alibaba-damo-academy/RynnMotion) <br>
> RynnBot Team <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnMotion)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnMotion.svg?style=social)](https://github.com/alibaba-damo-academy/RynnMotion)  <br>

> [**RynnEC: Bringing MLLMs into Embodied World**](https://github.com/alibaba-damo-academy/RynnEC) <br>
> Ronghao Dang*, Yuqian Yuan*, Yunxuan Mao*, Kehan Li*, Jiangpin Liu, Zhikai Wang, Fan Wang, Deli Zhao, Xin Li <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/alibaba-damo-academy/RynnEC)  [![github](https://img.shields.io/github/stars/alibaba-damo-academy/RynnEC.svg?style=social)](https://github.com/alibaba-damo-academy/RynnEC) [![arXiv](https://img.shields.io/badge/Arxiv-2508.14160-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.14160) <br>


</p></details>

## 📰 News

* **[2026.02.09]**  🔥🔥 Release our code and model checkpoints ！！



## Introduction
We present **RynnBrain**, an embodied foundation model grounded in physical reality. Rynnbrain is available in two dense variants (2B and 8B) and one mixture-of-experts (MoE) model (30B-A3B). 
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
RynnBrain employs a unified encoder-decoder architecture (supporting both Dense and MoE variants) to transform omni-vision inputs and textual instructions into multi-modal outputs, including spatial trajectories, physical pointing, and action planning. 
Through massive training on rich spatio-temporal, physical-space, and general knowledge data, RynnBrain maintains robust general-purpose capabilities while specializing in diverse, fine-grained embodied reasoning and complex planning tasks.

<p align="center">
<img src="https://github.com/alibaba-damo-academy/RynnBrain.github.io/blob/main/assets/frame.png?raw=true" style="width: 90%; height: auto;">
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
<img src="https://github.com/user-attachments/assets/488da393-a828-4bd9-8813-ace6164a1d2c" style="width: 80%; height: auto;">
</p>




- Vision-Language Navigation

<p align="center">
<img src="https://github.com/user-attachments/assets/78c36b4e-0ea8-42e2-a3fd-692d7c2fb4a7" style="width: 80%; height: auto;">
</p>





## Model Zoo

| Model            | Base Model           | Huggingface | Modelscope |
| :--------------- | :------------------- | :---------: | :--------: |
| RynnBrain-2B  | Qwen3-VL-2B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-2B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-2B)   |
| RynnBrain-8B  | Qwen3-VL-8B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-8B)   |
| RynnBrain-30B-A3B  | Qwen3-VL-30B-A3B-Instruct | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-30B-A3B)   |
| RynnBrain‑CoP-8B | RynnBrain-8B         | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-CoP-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-CoP-8B)   |
| RynnBrain‑Plan-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-8B)   |
| RynnBrain‑Plan-30B-A3B | RynnBrain-30B-A3B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Plan-30B-A3B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Plan-30B-A3B)   |
| RynnBrain‑Nav-8B | RynnBrain-8B        | [Link](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-Nav-8B)    | [Link](https://www.modelscope.cn/models/DAMO_Academy/RynnBrain-Nav-8B)   |



## Quick Start

Minimal dependencies:
```shell
pip install transformers==4.57.1
```
Run text generation:
```python
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("")
...
```


## Cookbooks
Checkout the [cookbooks](./cookbooks) that showcase RynnBrain's capabilities in cognition, localization, reasoning, and planning.


| Category             | Cookbook name                                                                                   | Description |
|----------------------|--------------------------------------------------------------------------------------------------|-------------|
| Cognition            | [1_spatial_understanding.ipynb](./cookbooks/1_spatial_understanding.ipynb)                     | Shows the ability of model for spaital understanding in the video scene. |
| Cognition            | [2_object_understanding.ipynb](./cookbooks/2_object_understanding.ipynb)                       | Shows how the model understands object categories, attributes, and relations and counting ability. |
| Cognition            | [3_ocr.ipynb](./cookbooks/3_ocr.ipynb)                                                         | Examples of optical character recognition and text understanding in videos. |
| Location             | [4_object_location.ipynb](./cookbooks/4_object_location.ipynb)                                 | Locates specific objects with bounding boxes in an image or video based on instructions. |
| Location             | [5_area_location.ipynb](./cookbooks/5_area_location.ipynb)                                     | Identifies and marks specified regions by points in an image or video. |
| Location             | [6_affordance_location.ipynb](./cookbooks/6_affordance_location.ipynb)                         | Finds areas or objects with specific affordances in an image of video. |
| Location             | [7_trajectory_location.ipynb](./cookbooks/7_trajectory_location.ipynb)                         | Infers and annotates trajectories or motion paths in an image or video. |
| Location             | [8_grasp_pose.ipynb](./cookbooks/8_grasp_pose.ipynb)                                           | Present the model's abiltiy to predict robotic grasp poses from images. |
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


## Acknowledgement

Our RynnBrain is built on top of [**Qwen3-VL**](https://github.com/QwenLM/Qwen3-VL). We also learned a lot from the implementation of [**RynnEC**](https://github.com/alibaba-damo-academy/RynnEC) and [**VideoRefer**](https://github.com/DAMO-NLP-SG/VideoRefer). If your work is used in RynnBrain but not mentioned in either this repo or the technical report, feel free to let us know :heart:.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
