from dotenv import load_dotenv
import argparse
import time
from processing import test_model

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("model", help = "find the accuracy of the given model(ex: gemini-2.5-flash)")
parser.add_argument("-i", "--iteration", help = "repeat i many times for more accurate result", type = int )
args = parser.parse_args()

# Data
doc_names =[
    'id_card_1.png',
    'id_card_2.png',
    'id_card_3.jpg',
    'id_card_4.png',
    'id_card_5.jpg',
]

ground_truth_names = [
    'id_card_1.json',
    'id_card_2.json',
    'id_card_3.json',
    'id_card_4.json',
    'id_card_5.json',
]

schema1 = "id_card.json"

m = args.model

start_time = time.perf_counter()

if args.iteration:
    repeat = args.iteration
    total_field_avg = 0
    total_char_avg = 0
    for i in range(repeat):
        avg = test_model(m, doc_names, ground_truth_names,schema1)
        total_field_avg += avg["field_acc"]
        total_char_avg += avg["char_acc"]
    avr_field_avg = total_field_avg/repeat
    avr_char_avg = total_char_avg/repeat

    print(f"{m} average field accuracy after {repeat} tries: ", avr_field_avg)
    print(f"{m} average character accuracy after {repeat} tries: ", avr_char_avg)

else:
    avg = test_model(m, doc_names, ground_truth_names,schema1)

    print(f"{m} average field accuracy: ", avg["field_acc"])
    print(f"{m} average character accuracy: ", avg["char_acc"])

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"{m} executed in {elapsed_time:.6f} seconds\n")

