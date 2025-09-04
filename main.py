import argparse
import time
from processing import test_model, process_csv

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
repeat = 1

if args.iteration:
    repeat = args.iteration
    total_field_avg = 0
    total_char_avg = 0
    for i in range(repeat):
        avg = test_model(m, doc_names, ground_truth_names,schema1)
        total_field_avg += avg["field_acc"]
        total_char_avg += avg["char_acc"]
    avg_field = total_field_avg/repeat
    avg_char = total_char_avg/repeat


    print(f"{m} average field accuracy after {repeat} tries: ", avg_field)
    print(f"{m} average character accuracy after {repeat} tries: ", avg_char)

else:
    avg = test_model(m, doc_names, ground_truth_names,schema1)
    avg_field = avg["field_acc"]
    avg_char = avg["char_acc"]

    print(f"{m} average field accuracy: ", avg_field)
    print(f"{m} average character accuracy: ", avg_char)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"{m} executed in {elapsed_time:.6f} seconds\n")

append_row = [m, repeat, round(avg_field,7), round(avg_char,7), round(elapsed_time,7)]
process_csv("results.csv", append_row)



