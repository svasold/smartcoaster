import csv
import numpy as np
import matplotlib.pyplot as plt


# Read Data
with open('/home/marco/Schreibtisch/TU/Bac/data_spec/data_spec.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    csv_data = np.array(list(reader))
unique, counts = np.unique(csv_data[:, -2], return_counts=True)
print(dict(zip(unique, counts)))
# data = csv_data[:, [0, 1, 2, 3, 4, 5, -3, -2]]
data = {}
water, beer, apple_juice, red_wine, water, white_wine = [], [], [], [], [], []
for csv_row in csv_data:
    if int(csv_row[-1]) == 0:
        drink = str(csv_row[-2])
        row = np.array(csv_row[[0, 1, 2, 3, 4, 5, -3]]).astype(float)
        if drink in data:
            data[drink] = np.vstack([data[drink], row])
        else:
            data[drink] = row
# water_weights = np.sort(data["water"][:, -1])
# water_ch1 = data["water"][:, 0]
# water_weight_ch1 = np.vstack([np.sort(data["water"][:, -1]), data["water"][:, 0]])

for drink in data:
    data[drink] = data[drink][data[drink][:, -1].argsort()]
    print(drink)
    print(data[drink][-1])
    plt.scatter(data[drink][:, -1], data[drink][:, 0], label=drink + " channel 1")
    plt.ylabel('ch1')
    plt.xlabel('weight')
    plt.legend()
    plt.show()
print("fertig")
