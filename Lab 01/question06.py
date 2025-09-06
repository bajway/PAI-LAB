# Given a dictionary of countries and their populations, write a program to find the
# top 3 most populated countries without using sorted().

Data= {
    "Pakistan": 250000000,
    "Nepal": 2000000,
    "India": 300000000,
    "Bhutan": 200000,
    "China": 450000000
}
copyData= dict(Data) #to store a copy of original data
for i in range(3):
    max_country = max(copyData, key=copyData.get)
    print(max_country, ":", copyData[max_country])
    copyData.pop(max_country)
