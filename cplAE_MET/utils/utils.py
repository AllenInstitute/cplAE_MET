import csv

def write_list_to_csv(path, file):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(file)
    print("Done writing!")

def read_list_from_csv(path):
    with open(path, 'r') as myfile:
        reader = csv.reader(myfile)
        data = list(reader)
    return data[0]
