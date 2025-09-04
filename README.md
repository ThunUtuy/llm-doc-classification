# llm-doc-classification
Evaluating which models are the fastest and most accurate at extracting data

## How to Run
1. Set up venv
```
python venv venv
venv/Scripts/activate
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Add in your own .env file
```
API_KEY = your_key
BASE_URL = your_base_url
```
4. Run the program
```
python main.py gemini-2.5-flash

# Run more than once with -i or --iteration for more accuracy
python main.py gemini-2.5-flash --iteration 2
```
5. Check the csv file for results (not done)
- The rows contain: model, repetition, field accuracy, character accuracy, time taken to run

6. Changing files to test
  - Change the files in the data/docs
  - Change the schemas in the data/schemas
  - Change the fields in the ground_truth according to the files in docs and schemas

