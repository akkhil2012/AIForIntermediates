

# Create fresh env
#python -m venv rag_env
#source rag_env/bin/activate       # Mac/Linux
#rag_env\Scripts\activate          # Windows


# Install everything clean
#pip install transformers torch sentence-transformers numpy huggingface_hub'''

from transformers import pipeline
from PIL import Image
import requests

# Load VQA pipeline
vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

# Load image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
image = Image.open(requests.get(image_url, stream=True).raw)

# Ask a question about the image
question = "What is in the image?"
result = vqa(image, question)
print("Answer:", result[0]["answer"])
