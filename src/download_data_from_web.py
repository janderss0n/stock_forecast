import requests
from bs4 import BeautifulSoup

URL = 'http://www.nasdaq.com/symbol/amd/historical'
page = requests.get(URL).text
soup = BeautifulSoup(page, 'lxml')
tableDiv = soup.find_all('div', id="historicalContainer")
tableRows = tableDiv[0].findAll('tr')

colNameRow=tableDiv[0].findAll('th')
columnNames = list()
for name in colNameRow:
    columnNames.append(name.getText().split()[0])
print(columnNames)


for tableRow in tableRows[2:]:
    row = list(tableRow.getText().split())
    print(row)
