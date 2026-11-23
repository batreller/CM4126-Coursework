# Oleksandr Reshetnikov (2409418)
# Streamlit app for final model (CM4126, ISIC 2019)
# it can be run after models/final_model.json is saved, to run it use streamlit run app.py


import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy import ndimage
from torchvision.io import decode_jpeg, encode_jpeg, read_file
from torchvision.models import inception_v3, resnet50
from torchvision.transforms import v2

IMAGE_SIZE = 256
CROP_SIZE = 224
TEST_DIR = Path("data/processed/isic2019/test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFO = json.loads(Path("models/final_model.json").read_text())  # which model was chosen by notebook


@st.cache_resource
def load_model():
    # same network as in notebook, weights from models folder
    if INFO["architecture"] == "ResNet50":
        model = resnet50(weights=None)
    else:
        model = inception_v3(weights=None, aux_logits=False, transform_input=False, init_weights=False)
    model.fc = nn.Linear(model.fc.in_features, len(INFO["classes"]))
    model.load_state_dict(torch.load(INFO["weights"], map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def shades_of_gray_tensor(image, p=6):
    # color constancy with same formula as in notebook
    pixels = image.float()
    light = pixels.pow(p).mean(dim=(1, 2)).pow(1 / p)
    light = light / light.norm()
    corrected = pixels / (light * 3 ** 0.5).view(3, 1, 1)
    return corrected.clamp(0, 255).to(torch.uint8)


HAIR_ANGLES = [k * np.pi / 8 for k in range(8)]


def line_closing(gray, half_length):
    # same as in notebook
    size = gray.shape[-1]
    results = []
    for angle in HAIR_ANGLES:
        steps = [(int(round(k * np.sin(angle))), int(round(k * np.cos(angle))))
                 for k in range(-half_length, half_length + 1)]
        image = gray
        for reduce in (torch.amax, torch.amin):
            padded = nn.functional.pad(image[None], [half_length] * 4, mode="replicate")[0]
            shifted = [padded[:, half_length + dy:half_length + dy + size, half_length + dx:half_length + dx + size]
                       for dy, dx in steps]
            image = reduce(torch.stack(shifted), dim=0)
        results.append(image)
    return torch.stack(results)


def remove_hair(image, threshold=15, min_length=30):
    # simplified DullRazor (same function as in notebook)
    pixels = image.float()
    gray = pixels.mean(dim=0, keepdim=True)
    thin = line_closing(gray, 3).amax(0) - gray
    along = line_closing(gray, 10).amin(0) - gray
    hair = ((thin > threshold) & (along < threshold / 2))[0].cpu().numpy()
    labels, count = ndimage.label(hair, structure=np.ones((3, 3)))
    long_piece = np.zeros(count + 1, dtype=bool)
    for number, (rows, cols) in enumerate(ndimage.find_objects(labels), start=1):
        long_piece[number] = max(rows.stop - rows.start, cols.stop - cols.start) >= min_length
    hair = torch.from_numpy(long_piece[labels]).to(image.device)[None, None].float()
    hair = nn.functional.max_pool2d(hair, 3, stride=1, padding=1)[0] > 0
    clean = (~hair).float()
    colour_sum = nn.functional.avg_pool2d((pixels * clean)[None], 11, stride=1, padding=5, count_include_pad=False)[0]
    clean_share = nn.functional.avg_pool2d(clean[None], 11, stride=1, padding=5, count_include_pad=False)[0]
    filled = colour_sum / clean_share.clamp(min=1e-6)
    result = torch.where(hair & (clean_share > 0.05), filled, pixels)
    return result.clamp(0, 255).to(torch.uint8), hair


def model_view(image):
    # if final model was trained on image without hair we remove hair and save new jpeg
    if not INFO.get("hair_removal", False):
        return image
    cleaned, hair = remove_hair(image)
    return decode_jpeg(encode_jpeg(cleaned.cpu(), quality=95)).to(DEVICE) if hair.any() else image


def prepare_image(file_bytes):
    # like resize_image in Pre-processing
    with Image.open(io.BytesIO(file_bytes)) as img:
        img.draft("RGB", (2 * IMAGE_SIZE, 2 * IMAGE_SIZE))
        square = ImageOps.fit(img.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    square.save(buffer, format="JPEG", quality=95)
    return decode_jpeg(torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)).to(DEVICE)


def grad_cam(model, image, layer):
    # Grad-CAM like in notebook
    saved = {}
    handle = layer.register_forward_hook(lambda module, inputs, output: saved.update(maps=output))
    scores = model(image.unsqueeze(0))
    handle.remove()
    predicted = scores.argmax(1).item()
    gradients = torch.autograd.grad(scores[0, predicted], saved["maps"])[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * saved["maps"]).sum(dim=1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=image.shape[1:], mode="bilinear")
    heatmap = (cam[0, 0] / (cam.max() + 1e-8)).detach().cpu().numpy()
    return heatmap, scores.softmax(1)[0].detach().cpu().numpy()


# image processing for the model (color constancy, 224x224 crop, normalization)
steps = [shades_of_gray_tensor] if INFO["colour_constancy"] else []
transform = v2.Compose([*steps, v2.CenterCrop(CROP_SIZE), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(INFO["norm_mean"], INFO["norm_std"])])

st.title("Skin lesion classification (ISIC 2019)")
st.write(f"Final model of the coursework: **{INFO['name']}**. "
         "This is a student project and must not be used for medical decisions.")

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])
if TEST_DIR.exists() and st.button("Or take a random image from the test set"):
    st.session_state["example"] = pd.read_csv("splits/test.csv").sample(1).iloc[0].to_dict()

true_class = None
if uploaded is not None:
    image = prepare_image(uploaded.getvalue())
elif "example" in st.session_state:
    example = st.session_state["example"]
    image = decode_jpeg(read_file(str(TEST_DIR / f"{example['image']}.jpg"))).to(DEVICE)  # already 256x256
    true_class = example["label"]
else:
    st.stop()

model = load_model()
layer = model.layer4 if hasattr(model, "layer4") else model.Mixed_7c
heatmap, probs = grad_cam(model, transform(model_view(image)), layer)

# image how it's seen by human and Grad-CAM on top
picture = v2.CenterCrop(CROP_SIZE)(image).permute(1, 2, 0).cpu().numpy() / 255
overlay = 0.6 * picture + 0.4 * matplotlib.colormaps["jet"](heatmap)[..., :3]
left, right = st.columns(2)
left.image(picture, caption="Image (centre 224x224)", width="stretch")
right.image(overlay, caption="Grad-CAM: regions that influenced the prediction", width="stretch", clamp=True)

predicted = int(probs.argmax())
st.subheader(f"Prediction: {INFO['classes'][predicted]} ({INFO['class_names'][predicted]}), "
             f"probability {probs[predicted]:.2f}")
if true_class is not None:
    st.write(f"True class of this test image: **{true_class}**")
labels = [f"{c} ({name})" for c, name in zip(INFO["classes"], INFO["class_names"])]
st.bar_chart(pd.DataFrame({"Probability": probs}, index=labels), horizontal=True, sort=False)
