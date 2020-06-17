from collections import OrderedDict
import csv
import os
import sys

def normalize_file(filename, evt_name_column=True):
    """Normalizes probabilities in a file. Modifies the original file."""
    total_prob = 0
    terms = OrderedDict()
    count = 0
    try:
        with open(filename, "r") as prob_file:
            reader = csv.reader(prob_file)
            for row in reader:
                terms[row[0]] = float(row[1])
                
    except IOError:
        print("Error reading to or opening file")
        sys.exit(1)

    normalized_dict = normalize_dict(terms)

    try:
        with open(filename, "w") as prob_file:
            writer = csv.writer(prob_file)
            for term in normalized_dict:
                writer.writerow([term, normalized_dict[term]])
    except IOError:
        print("Error writing to or opening file")
        sys.exit(1)

def normalize_dict(evt_dict):
    """Normalizes probabilities of events in a dictionary. Returns a new normalized dictionary."""
    total_prob = 0
    new_dict = {}
    for evt in evt_dict:
        total_prob += evt_dict[evt]
    for evt in evt_dict:
        new_dict[evt] = evt_dict[evt] / total_prob
    
    return new_dict

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: normalize.py filename")
        sys.exit(1)

    filename = sys.argv[1]
    if not os.path.exists(filename):
        print("File does not exist. Check to make sure filepath is correct.")
        sys.exit(1)
    normalize_file(filename)
