from openai import OpenAI
from PIL import Image
import numpy as np
import json
import base64
import jsonschema
import metrics

def get_page_extraction(page_image_path: str, json_schema_str: str) -> dict:
    """
Extracts structured data and raw text from a page image.

- If a schema_id is provided, it extracts structured data according to the schema
    and also extracts the raw text. It returns a dict with 'parsed_json' and 'raw_text'.
    """
    client = OpenAI(
        #Mfec Key
        #base_url="https://gpt.mfec.co.th/litellm",
        #api_key="sk-5quRoBB531T6nRkbNPBsGg",

        #My key
        api_key="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"
    )

    try:
        with open(page_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {page_image_path}")
        return None
    
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }

    prompt = f"""Extract all text from the document and structured data based on the provided JSON schema.
Return a single JSON object with two top-level keys:
1. `raw_text`: A string containing all the extracted text from the document in markdown format.
2. `extracted_data`: A JSON object containing the structured data that conforms to the schema below.

JSON Schema:
```json
{json_schema_str}
```"""
    messages = [
        {
            "role": "system",
            "content": "You are an expert data extractor. You must return data in the exact JSON format requested, with 'raw_text' and 'extracted_data' keys.",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, image_content],
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )

    try:
        response_json = json.loads(response.choices[0].message.content)

        # Structured extraction validation
        raw_text = response_json.get("raw_text")
        extracted_data = response_json.get("extracted_data")

        return {"parsed_json": extracted_data}

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        return None
    except jsonschema.exceptions.ValidationError as e:
        print(f"LLM response failed schema validation: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return None
    
def extract_file(schema_name, doc_name):
    try:
        with open(f'data/schemas/{schema_name}', 'r') as file:
            schema_data = json.load(file)
        
    except FileNotFoundError:
        print("Error: The file 'data.json' was not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from the file.")
        return

    ans = get_page_extraction(f"data/docs/{doc_name}", schema_data)
    return ans["parsed_json"]

def convert_np_array(json_str:str):
    ans = list(json_str.keys())
    truth = np.array([json_str[f] for f in ans])
    return truth

def get_truth():
    #Get Truth From labels.json
    with open("ground_truth/id_card_1.json") as f:
        labels = json.load(f)

    filenames = list(labels.keys())
    truth = np.array([labels[f] for f in filenames])
    return truth


ans = extract_file("id_card.json", "id_card_1.png")
preds = convert_np_array(ans)
print(preds)

print(get_truth())
truth = get_truth()

acc = metrics.accuracy(truth, preds)
print(acc)
        

