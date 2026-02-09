# RynnBrain-Bench
<p align="center">
<img src="../cookbooks/assets/logo.png" style="width: 30%; height: auto;">
</p>

## Introduction


We introduce **RynnBrain-Bench**, a high-dimensional evaluation suite designed to holistically benchmark the cognition and localization capabilities of embodied understanding models in complex household environments.
Advancing beyond existing benchmarks, RynnBrain-Bench features a unique emphasis on fine-grained understanding and precise spatiotemporal localization within episodic video sequences.

RynnBrain-Bench systematically measures spatiotemporal embodied understanding across four foundational pillars: *Object Cognition*, *Spatial Cognition*, *Grounding*, and *Pointing*. Covering 21 specialized sub-capabilities ranging from detailed object attributes to affordance points prediction, the benchmark comprises 3,616 video clips and 8,000 meticulously curated open-ended questions, serving as a rigorous evaluation ground for next-generation multimodal models.

<p align="center">
<img src="../cookbooks/assets/RynnBrain-Bench.png" style="width: 80%; height: auto;">
</p>

## Evaluation Dimensions
### *1. Object Cognition*
Object Cognition challenges models with fine-grained perception and object counting of region-level targets across dynamic video sequences. We assess nine core attributes—such as state, size, and surface detail—plus a distinct object counting capability. To ensure high fidelity, all questions are human-verified and balanced against real-world object distributions for maximum authenticity.

*Evaluation Metrics:* (1) Binary or fine-grained scores from GPT-4o.

### *2. Spatial Cognition*
Spatial cognition requires MLLMs to derive 3D spatial awareness from ego-centric video streams, spanning two primary perspectives: Ego-centric and World-centric. While ego-centric cognition examines the model's evolving relationship with its environment over time, world-centric cognition evaluates the comprehension of objective 3D layouts and physical properties, such as scale, distance, and position.

*Evaluation Metrics:*  (1) Neumerical Questions: Mean Relative Accuracy (MRA) and Rotational Accuracy (RoA) (2)Textual Questions: Binary or fine-grained scores from GPT-4o.
### *3. Grounding*
Grounding evaluates the capability for precise spatiotemporal localization, representing a key link for anchoring understanding in reality. This task requires the brain model to (1) pinpoint the critical temporal keyframe and then (2) predict the object's spatial coordinates within that frame. We distinguish between Direct Grounding, which involves locating objects based on explicit descriptions, and Situational Grounding, which necessitates context-aware reasoning to identify and localize targets within complex scenarios.

*Evaluation Metrics:* (1) Accuracy@0.5
### *4. Pointing*
The Pointing task aims to predict target areas, spatio-temporal trajectories, or affordance points across the entire episodic memory, serving as a critical bridge for robot-physical world interaction. Departing from previous benchmarks, we extend the evaluation scope to the spatiotemporal domain, where models must demonstrate the dual capacity to locate the key frame and predict corresponding task-relevant point sequences.

*Evaluation Metrics:* (1) Area: Proportion of prediction points falling within the target area (2) Affordance: Euclidean distance (3) Trajectory: Discrete Fréchet Distance (DFD)

## Leaderborad

---
| Model | Object Cognition | Spatial Cognition | Grounding | Area | Affordance | Trajectory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 41.8 | 35.0 | 62.8 | 30.0 | 82.9 | 63.4 |
| Cosmos-reason2-8B | 37.2 | 31.4| 60.0 | 37.6 | 83.9 | 64.0 |
| RoboBrain-2.0-7B |24.7 | 13.5 | 18.6 | 38.0 | 73.5 | 57.6 |
| Pelican-VL-7B | 30.8 | 20.5 | 3.5 | 46.5 | 81.4 | 59.2 |
| MiMo-Embodied-7B | 39.0 | 28.3 | 49.8 | 49.4 | 84,4 | 61.3 |
| **RynnBrain-2B** | **70.7** | **57.2** | **79.1** | **54.6** | **89.4** | **66.6** |
| **RynnBrain-8B** | **71.2** | **59.9** | **81.6** | **56.2** | **90.4** | **64.5** |
---
| Model | Object Cognition | Spatial Cognition | Grounding | Area | Affordance | Trajectory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-5.2 | 55.0 | 36.4 | 14.9 | 33.9 | 83.1 | 70.4 |
| Genimi-3 Pro | 58.4 | 38.5 | 62.1 | 61.5 | 86.0 | 72.0 |
| Claude Sonnet 4.5 | 25.1 | 34.2 | 0.0 | 10.1 | 38.7 | 54.6 |
| Qwen3-VL-30B-A3B | 42.6 | 30.7 | 76.4 | 30.9 | 86.2 | 61.2 |
| RoboBrain-2.0-30B | 26.2 | 11.6 | 0.0 | 45.3 | 76.1 | 60.3 |
| Pelican-VL-72B | 42.2 | 32.2 | 10.8 | 53.2 | 87.3 | 64.1 |
| **RynnBrain-30B-A3B** | **73.3** | **59.3** | **83.9** | **59.3** | **90.5** | **66.8** |
---

## Data Format
```json
{
  "id": 3041,
  "conversation": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "e7758931b99a1e5a8140ba5811d64863/00000.jpg",
        },
        {
          "type": "image",
          "image": "e7758931b99a1e5a8140ba5811d64863/00015.jpg",
        },
        {
          "type": "text",
          "text": "What category does the <object> <frame53>; (333, 430), (453, 651) </object> belong to?"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "The object is a stuffed toy or plush bear."
        }
      ]
    }
  ],
  "data_source": "object_cognition",
  "task_type": "category"
}
```
* `id`: Unique identifier for each data entry.
* `conversation`: Image paths, Question, and Ground Truth.
* `data_source`: The general category of the task.
* `task_type`: The specific category of the task.

## Data Download
The data and jsonl files of RynnBrain-Bench can be downloaded [here](https://huggingface.co/datasets/Alibaba-DAMO-Academy/RynnBrain-Bench). You need to first unzip the `data.zip` file.

Data structure:
```bash
RynnBrain-Bench
├── data
│   └──... (videos)
└── json
    └── rynnbrain_object_2000.jsonl
    └── rynnbrain_spatial_2000.jsonl
    └── ...

```

## Evaluation
Please refer to [RynnScale](https://github.com/alibaba-damo-academy/RynnScale#evaluation) for details.
```bash
python -m rynn_scale.api.eval \
    --model_path $MODEL_PATH \
    --benchmarks RynnBrainCog RynnBrainLoc \
    --prompt_format RynnBrain \
    --save_dir $SAVE_DIR \
    --backend hf \
    --num_processor_workers 4 \
    --fps 2 \
    --max_frames 512 \
    --image_min_pixels $((16 * 32 * 32)) \
    --image_max_pixels $((16384 * 32 * 32)) \
    --video_max_pixels $((24576 * 32 * 32 * 2)) \
    --temperature 0.0
```

