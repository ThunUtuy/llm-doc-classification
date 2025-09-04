from openai import OpenAI, BadRequestError
import numpy as np
import json
import base64
import jsonschema
import metrics
import os
import csv
from dotenv import load_dotenv
from Levenshtein import distance


load_dotenv()

def get_page_extraction(page_image_path: str, json_schema_str: str, model_name: str) -> dict:
    """
Extracts structured data and raw text from a page image.

- If a schema_id is provided, it extracts structured data according to the schema
    and also extracts the raw text. It returns a dict with 'parsed_json' and 'raw_text'.
    """
    client = OpenAI(
        #Mfec Key
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
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
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )

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
    except BadRequestError as e:
        print(f"Error requesting with the model - {model_name}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return None
    
def extract_file(schema_name, doc_name, model_name):
    """
    Return a json of the response, by inputting Schema and Document
    """
    try:
        with open(f'data/schemas/{schema_name}', 'r') as file:
            schema_data = json.load(file)
        
    except FileNotFoundError:
        print("Error: The file 'data.json' was not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from the file.")
        return

    ans = get_page_extraction(f"data/docs/{doc_name}", schema_data, model_name)
    return ans["parsed_json"]

def convert_np_array(json_str:str):
    """
    Convert a json answer into an numpy array ex. ['Mr. John' 'Doe']
    """
    ans = list(json_str.keys())
    truth = np.array([json_str[f] for f in ans])
    return truth


def get_truth(fname: str):
    """
    return a numpy array of values of the answers ex. ['Mr. John' 'Doe']
    """
    #Get Truth From labels.json
    with open(f"ground_truth/{fname}") as f:
        labels = json.load(f)

    keys = list(labels.keys())
    truth = np.array([labels[key] for key in keys])
    return truth

def test_model(model_name: str, train, train_ans, schema):
    total_accuracy = 0
    total_cer = 0
    length = len(train)
    for i in range(length):
        ans = extract_file(schema, train[i], model_name)
        preds = convert_np_array(ans)
        #print(preds)
        truth = get_truth(train_ans[i])
        #print(truth)

        """Field Accuracy"""
        acc = metrics.accuracy(truth, preds)
        #print(acc)
        total_accuracy += acc

        """Character Accuracy"""
        cer = calculate_cer(truth, preds)
        total_cer += cer
        #print(1 - cer)

    field_acc = total_accuracy / length
    character_acc = 1 - (total_cer/length)
    #print(f"{model_name} average field accuracy: ", field_acc, "")
    #print(f"{model_name} average character accuracy: ", character_acc, "\n")
    return {"field_acc":field_acc, "char_acc":character_acc}

def calculate_cer(reference, hypothesis):
    ref = [x for x in reference if x is not None]
    hyp = [x for x in hypothesis if x is not None]


    r = ''.join(ref)
    h = ''.join(hyp)

    # Calculate Levenshtein distance
    edit_distance = distance(r, h)
    # Calculate CER
    cer = edit_distance / len(r)
    return cer

def process_csv(file_name, content):
    try:
        with open(file_name, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(content)

    except FileNotFoundError as e:
        print(f"Error: The file '{file_name}' was not found.")
        return
    except Exception as e:
        print(e)
        return 

