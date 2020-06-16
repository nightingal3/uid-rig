import csv
import os
import sys

def average_need_probabilities(need_prob_list):
    np_list_length = len(need_prob_list)
    averaged_nps = float(sum(need_prob_list))/np_list_length
    return averaged_nps

def read_need_prob_csvs_and_avg(filenames_list, mapping):
    terms_prob_dict = {}
    for filename in filenames_list:
        print(filename)
        if os.path.exists(filename):
            try:    
                with open(filename, "r") as np_file:
                    reader = csv.reader(np_file)
                    print(filename)
                    for row in reader:
                        print(row)
                        if mapping[row[0]] in terms_prob_dict:
                            terms_prob_dict[mapping[row[0]]].append(float(row[1]))
                        else:
                            terms_prob_dict[mapping[row[0]]] = [float(row[1])]
            except IOError:
                print("Could not open or read from file " + filename)
                sys.exit(1)
        else:
            print("File does not exist. Check filepath.")
    for term in terms_prob_dict:
        if len(terms_prob_dict[term]) > 1:
            terms_prob_dict[term] = average_need_probabilities(terms_prob_dict[term])
        else:
            terms_prob_dict[term] = terms_prob_dict[term][0]
    print(terms_prob_dict)
    return terms_prob_dict

def read_mapping_files(terms_dir, language_names="all"):
    if language_names != "all":
        language_names = [name[0].upper() + name[1:] for name in language_names]
    print(language_names)
    mapping = {}
    for filename in os.listdir(terms_dir):
        if filename.endswith(".csv") and (language_names == "all" or any(name in filename for name in language_names)):
            full_filename = os.path.join(terms_dir, filename)
            print(full_filename)
            try:
                with open(full_filename, "r") as terms_file:
                    reader = csv.reader(terms_file)
                    for row in reader:
                        english_term, other_lang_term = row[0], row[1]
                        mapping[other_lang_term] = english_term
            except IOError:
                print("Could not open or write to file " + filename + " ...aborting.")
                sys.exit(1)
    print("mapping")
    print(mapping)
    return mapping


def write_need_probs(avg_need_probs_dict, filename="data/need_probs/universal_np.csv"):
    try:
        with open(filename, "w") as uni_np_file:
            writer = csv.writer(uni_np_file)
            for term in avg_need_probs_dict:
                writer.writerow([term, avg_need_probs_dict[term]])
    except IOError:
        print("Could not open or write to file " + filename)
        sys.exit(1)


def average_need_probs_and_write_to_file(destination_filename, target_filename_list, other_lang_to_eng_mapping):
    num_targets = len(locals()) - 1
    existing_target_files = []
    for target_filename in target_filename_list:
        if os.path.exists(target_filename):
            existing_target_files.append(target_filename)
        else:
            print("File " + target_filename + " not found. Skipping...")
    terms_prob_dict = read_need_prob_csvs_and_avg(existing_target_files, other_lang_to_eng_mapping)

    write_need_probs(terms_prob_dict, filename=destination_filename)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: average_need_probabilities.py destination_filename target_filename_0....target_filename_n")
        sys.exit(1)
    mapping = read_mapping_files("../data/kinship_terms")
    average_need_probs_and_write_to_file(sys.argv[1], [sys.argv[i] for i in range(2, len(sys.argv))], mapping)
    print("New probabilities written to " + sys.argv[1])

            


