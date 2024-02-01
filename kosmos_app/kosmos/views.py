from django.shortcuts import render

from django.http import HttpResponse
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image, ImageDraw
import requests

def index(request):
    # Load KOSMOS-2 model and processor
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224", load_in_4bit=True, device="cpu")
    #model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224", load_in_4bit=True, device_map={"": 0})

    # Load image from URL
    image_url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Prepare inputs for the model
    prompt = "<grounding>An image of"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

    # Autoregressively generate completion
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract entities
    processed_text, entities = processor.post_process_generation(generated_text)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for entity, _, box in entities:
        box = [round(i, 2) for i in box[0]]
        x1, y1, x2, y2 = tuple(box)
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
        draw.text(xy=(x1, y1), text=entity)

    # Save the annotated image
    annotated_image_path = "kosmos_app/static/kosmos_app/images/annotated_image.png"
    image.save(annotated_image_path)

    # Render the result in HTML
    return render(request, 'kosmos_app/index.html', {'annotated_image_path': annotated_image_path})
