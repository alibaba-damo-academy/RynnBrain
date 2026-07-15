import ffmpeg
import gradio as gr
import numpy as np
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor


FPS = 2
MAX_FRAMES = 512
MODEL_PATH = ""

if torch.xpu.is_available():
    DEVICE = "xpu:0"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},
    attn_implementation="sdpa",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)


INSTRUCTIONS = """
# Rynn Brain Demo

## Supported Visual Prompt Types
- **Object**: Define objects using two points (top-left and bottom-right of bounding box).
- **Area**: Define areas using multiple points.

## Usage Instructions
1. Upload an image or video.
2. Add text prompt.

[Optional] Add visual prompts by:

    3.1. Select the frame index (for videos).

    3.2. Choose the prompt type (object or area).

    3.3. Click on the image to select points. The selected point will be highlighted in green.

    3.4. Use "Add Point" to add the selected point. The added points will be highlighted in red.

    3.5. Use "Add Point Prompt" to add a textual prompt for the selected points.

4. Adjust sampling parameters if needed.
5. Click "Submit" to get the model's response.

## Note
- For 'object' prompts, ensure exactly two points are selected to define the bounding box.
- The model processes frames at 2 FPS for videos.
"""


def on_frame_idx_change(frames, frame_idx):
    image = Image.fromarray(frames[frame_idx].copy())
    return image, [], None


def on_image_upload(image):
    return np.array(image), gr.update(value=0, maximum=0, interactive=True), False, image, [], None


def get_frame(frames, frame_idx):
    if frames.ndim == 4:
        frame = frames[frame_idx]
    else:
        frame = frames
    frame = Image.fromarray(frame.copy())
    return frame


def draw_points_(frame, prompt_type, points, selected_point=None):
    if selected_point is not None:
        points = points + selected_point

    if len(points) == 0:
        return frame

    points = np.array(points)
    points[:, 0] = points[:, 0] / 1000 * frame.width
    points[:, 1] = points[:, 1] / 1000 * frame.height
    points = np.round(points).astype(int).tolist()

    draw = ImageDraw.Draw(frame)

    for i, point in enumerate(points):
        color = "green" if i == len(points) - 1 and selected_point is not None else "red"
        radius = max(max(frame.width, frame.height) // 75, 2)
        draw.ellipse([point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius], fill=color)

    if prompt_type == "object" and len(points) > 2:
        raise gr.Error("For 'object' prompt type, please select exactly two points to define the bounding box.")

    if prompt_type == "object" and len(points) == 2:
        if points[0][0] > points[1][0] or points[0][1] > points[1][1]:
            raise gr.Error("For 'object' prompt type, the first point should be the top-left and the second point the bottom-right of the bounding box.")
        width = max(max(frame.width, frame.height) // 100, 1)
        draw.rectangle(points[0] + points[1], outline="red", width=width)

    return frame


def on_image_select(frames, frame_idx, prompt_type, points, evt: gr.SelectData):
    frame = get_frame(frames, frame_idx)

    if prompt_type == "object" and len(points) == 2:
        gr.Warning("For 'object' prompt type, please select exactly two points to define the bounding box.")
        selected_point = None
    else:
        x, y = evt.index
        selected_point = [(round(x / frame.width * 1000), round(y / frame.height * 1000))]

    frame = draw_points_(frame, prompt_type, points, selected_point)
    return frame, selected_point


def on_prompt_type_change(frames, frame_idx, prompt_type, points, selected_point):
    frame = get_frame(frames, frame_idx)
    if prompt_type == "object":
        if len(points) > 2:
            gr.Warning("For 'object' prompt type, please select exactly two points to define the bounding box.")
            points = points[:2]
        if len(points) == 2 and selected_point is not None:
            gr.Warning("For 'object' prompt type, please select exactly two points to define the bounding box.")
            selected_point = None
        if len(points) == 2 and points[0][0] > points[1][0] or points[0][1] > points[1][1]:
            gr.Warning("For 'object' prompt type, the first point should be the top-left and the second point the bottom-right of the bounding box.")
            points = []
            selected_point = None
    frame = draw_points_(frame, prompt_type, points, selected_point)
    return frame, points, selected_point


def add_point(frames, frame_idx, prompt_type, points, selected_point):
    if selected_point is None:
        gr.Warning("Please select a point to add.")
    else:
        points = points + selected_point

    frame = get_frame(frames, frame_idx)
    frame = draw_points_(frame, prompt_type, points)

    return frame, points


def add_point_prompt(frames, frame_idx, prompt_type, points, text):
    if len(points) == 0:
        raise gr.Error("Please add at least one point before adding a point prompt.")

    if prompt_type == "object" and len(points) != 2:
        raise gr.Error("For 'object' prompt type, please add exactly two points to define the bounding box.")

    frame = get_frame(frames, frame_idx)
    frame = draw_points_(frame, prompt_type, [])

    frame_str = ""
    if frames.ndim == 4:
        frame_str = f"<frame {frame_idx}>;"

    points_str = ",".join([f"({point[0]},{point[1]})" for point in points])
    if len(text):
        text += " "
    text += f"<{prompt_type}> {frame_str} {points_str} </{prompt_type}>"

    return frame, text, [], None


def on_video_upload(video):
    probe = ffmpeg.probe(video)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    out, _ = (
        ffmpeg
        .input(video)
        .filter("fps", fps=FPS)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )

    video_array = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return (video_array, gr.update(value=0, maximum=len(video_array) - 1, interactive=True), False, *on_frame_idx_change(video_array, 0))


def visualize_point_prompts(frames, text):
    tags = ['object', 'area', 'affordance', 'trajectory']
    tag_pattern = '|'.join(tags)
    pattern = rf'(<(?:{tag_pattern})>.*?</(?:{tag_pattern})>)'
    main_pattern = rf'<({tag_pattern})>(.*?)</\1>'
    frame_pattern = r'<frame\s+(\d+)>;\s*(.*)'

    contents = []
    for chunk in re.split(pattern, text, flags=re.DOTALL):
        if len(chunk) == 0:
            continue

        match = re.match(main_pattern, chunk, flags=re.DOTALL)

        if match:
            prompt_type = match.group(1)
            contents.append(f"[{prompt_type}]: ")

            inner_content = match.group(2).strip()
            frame_match = re.search(frame_pattern, inner_content, flags=re.DOTALL)

            if frame_match:
                frame_idx = int(frame_match.group(1))
                content = frame_match.group(2).strip()
            else:
                frame_idx = 0
                content = inner_content

            point_pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
            points = [(int(point[0]), int(point[1])) for point in re.findall(point_pattern, content)]

            frame = get_frame(frames, frame_idx)
            frame = draw_points_(frame, prompt_type, points)
            contents.append(gr.Image(frame, type="pil"))

        else:
            contents.append(chunk)

    return [x.replace("<", "[").replace(">", "]") if isinstance(x, str) else x for x in contents]


@torch.inference_mode()
def submit(frames, is_frames_added, text, temperature, top_p, top_k, chatbot, conversation):
    contents = []
    chatbot_contents = visualize_point_prompts(frames, text)

    if frames is not None and not is_frames_added:
        if frames.ndim == 4:
            for frame_idx, frame in enumerate(frames):
                contents.append({"type": text, "text": f"<frame {frame_idx}>:"})
                contents.append({"type": "image", "image": Image.fromarray(frame)})
            chatbot_contents = ["[video]: ", gr.Image(frames[0])] + chatbot_contents
        elif frames.ndim == 3:
            contents.append({"type": "image", "image": Image.fromarray(frames)})
            chatbot_contents = ["[image]: ", gr.Image(frames)] + chatbot_contents
        else:
            raise gr.Error("Invalid frames dimension.")

    contents.append({"type": "text", "text": text})
    conversation.append({"role": "user", "content": contents})
    chatbot.append(gr.ChatMessage(role="user", content=chatbot_contents))

    model_inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs.to(model.device)

    if temperature == 0.0:
        sampling_parms = {"do_sample": False}
    else:
        sampling_parms = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

    output_ids = model.generate(
        **model_inputs,
        **sampling_parms,
        max_new_tokens=128,
    )
    response = processor.decode(output_ids[0, model_inputs["input_ids"].size(1):], skip_special_tokens=True)

    chatbot.append(gr.ChatMessage(role="assistant", content=visualize_point_prompts(frames, response)))
    conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

    return True, chatbot, conversation, ""


def main():
    with gr.Blocks() as demo:
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(height=800)
                    conversation = gr.State([])

                with gr.Column(scale=1):
                    with gr.Row():
                        image = gr.Image(type="pil", label="Upload Image")
                        video = gr.Video(label="Upload Video")
                        frames = gr.State()
                        is_frames_added = gr.State(False)

                    with gr.Row():
                        selected_point = gr.State(None)
                        points = gr.State([])
                        frame_idx = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Frame Index")
                        prompt_type = gr.Dropdown(choices=["object", "area"], value="object", label="Prompt Type", interactive=True)

                        with gr.Row():
                            add_point_btn = gr.Button("Add Point")
                            add_point_prompt_btn = gr.Button("Add Point Prompt")

                    text = gr.Textbox(label="Text Prompt", value="")

                    with gr.Accordion("Sampling Parameters", open=False):
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Temperature")
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.01, label="Top-p")
                        top_k = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top-k")

                    submit_btn = gr.Button("Submit")

        with gr.Tab("Instructions"):
            gr.Markdown(INSTRUCTIONS)

        image.upload(on_image_upload, inputs=[image], outputs=[frames, frame_idx, is_frames_added, image, points, selected_point])
        image.select(on_image_select, inputs=[frames, frame_idx, prompt_type, points], outputs=[image, selected_point])

        video.upload(on_video_upload, inputs=[video], outputs=[frames, frame_idx, is_frames_added, image, points, selected_point])

        frame_idx.change(on_frame_idx_change, inputs=[frames, frame_idx], outputs=[image, points, selected_point])
        prompt_type.change(on_prompt_type_change, inputs=[frames, frame_idx, prompt_type, points, selected_point], outputs=[image, points, selected_point])

        add_point_btn.click(add_point, inputs=[frames, frame_idx, prompt_type, points, selected_point], outputs=[image, points])
        add_point_prompt_btn.click(add_point_prompt, inputs=[frames, frame_idx, prompt_type, points, text], outputs=[image, text, points, selected_point])

        submit_btn.click(submit, inputs=[frames, is_frames_added, text, temperature, top_p, top_k, chatbot, conversation], outputs=[is_frames_added, chatbot, conversation, text])

    demo.launch(server_name="0.0.0.0", server_port=8060)


if __name__ == "__main__":
    main()
