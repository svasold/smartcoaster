import csv


file = open("/home/svasold/1_Uni/Bachelorarbeit/smartcoaster/smartcoaster/data_spec/data_final.csv", "r")
data = list(csv.reader(file, delimiter=","))
print(str(len(data)) + " " + str(len(data[0])))

new_deriv = True
drink = ""
container = ""
level = 0

deriv_data = []

deriv_counter = 0

for row in data:
    if new_deriv:
        drink = row[0]
        level = int(row[1])
        container = row[2]
        
        print(container + " " + drink + " " + str(level))
        new_deriv = False
        deriv_counter += 1

    if drink != row[0] or container != row[2] or int(row[1]) > level + 20:
        if (drink != "beer" and len(deriv_data) != 50):
            print("something is wrong")
            print(len(deriv_data))
            break
        deriv_data =[]
        new_deriv = True
    
    deriv_data.append(row)
print(deriv_counter)