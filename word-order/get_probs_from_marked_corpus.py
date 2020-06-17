import csv
import pickle

def get_all_events_and_probs(filepath):
    evt_occurrences = {}
    try:
        with open(filepath, "r") as probs_file:
            reader = csv.reader(probs_file, delimiter=" ")
            for line in reader:
                evt = tuple(line)
                if evt in evt_occurrences:
                    evt_occurrences[evt] += 1
                else:
                    evt_occurrences[evt] = 1
    except IOError:
        print("Error opening or reading file\n")
    

    return evt_occurrences

def write_to_evt_prob_file(filepath, evt_occurrences_dict):
    try:
        with open(filepath, "w") as probs_file:
            writer = csv.writer(probs_file, delimiter=",")
            for evt in evt_occurrences_dict:
                writer.writerow([evt_occurrences_dict[evt], evt[0], evt[1], evt[2]])
    except IOError:
        print("Error opening or writing to file\n")

def normalize_dict(evt_dict):
    """Normalizes probabilities of events in a dictionary. Returns a new normalized dictionary."""
    total_prob = 0
    new_dict = {}
    for evt in evt_dict:
        total_prob += evt_dict[evt]
    for evt in evt_dict:
        new_dict[evt] = float(evt_dict[evt]) / total_prob
    
    return new_dict


if __name__ == "__main__":
    probs = normalize_dict(get_all_events_and_probs("svo_corpus.csv"))
    with open("corpus_event_probs.p", "wb") as f:
        pickle.dump(probs, f)
    #write_to_evt_prob_file("corpus_event_probabilities.csv", probs)