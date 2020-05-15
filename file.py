import json

# my_details = {
#     'name': 'John Doe',
#     'age': 29
# }
#
# with open('personal.json', 'w') as json_file:
#     json.dump(my_details, json_file)


with open('data.json') as json_file:
    data = json.load(json_file)

    for x in data:
        # print(type(data[x]))
        # print(data[x])
        for y in data[x]:
            # print('ind')
            # print(y)
            for z in y:
                print(z)
