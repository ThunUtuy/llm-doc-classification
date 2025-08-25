from openai import OpenAI
from PIL import Image
import numpy as np
import json
import base64
import metrics

# Set the base URL for OpenAI to your custom endpoint
def get_doc_type():

    #Get Truth From labels.json
    with open("labels.json") as f:
        labels = json.load(f)

    filenames = list(labels.keys())

    client = OpenAI(
        #Mfec Key
        #base_url="https://gpt.mfec.co.th/litellm",
        #api_key="sk-5quRoBB531T6nRkbNPBsGg",

        #My key
        api_key="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"
    )

    raw = []

    for file in filenames:
        # Example: Create a chat completion
        with open(f"data/{file}", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages= [
                #{"role": "system", "content": "You are a document classifier. Return JSON only: { \"label\": <one of [\"national_id\",\"passport\",\"driver_license\",\"other\"]>, \"confidence\": <0..1> }. No extra text."},
                {"role": "system", "content": "You are a document classifier. Return just one of the words: national_id, passport, driver_license, other . No extra text."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify this document."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},}
                ]}  
            ]
        )
        raw.append(response.choices[0].message.content)
        
    preds = np.array(raw)
    print(preds)  
    return preds

def get_truth():
    #Get Truth From labels.json
    with open("labels.json") as f:
        labels = json.load(f)

    filenames = list(labels.keys())
    truth = np.array([labels[f] for f in filenames])
    return truth





truth = get_truth()
preds = get_doc_type()
acc = metrics.accuracy(truth, preds)
avg_prec = metrics.avg_precision
print("accuracy: ", acc)
print("avg_prec ", avg_prec)
