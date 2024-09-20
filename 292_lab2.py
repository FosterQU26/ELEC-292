import requests
from bs4 import BeautifulSoup

http_text = requests.get("https://weather.com/en-CA/weather/tenday/l/584018bec07ce9573837c14fa59da031fa6fcdeb1c3c9e3b2b27cb79ce254b5a#detailIndex5").text

#print(http_text)

soup = BeautifulSoup(http_text, 'lxml')
weather_data = soup.find_all('div', class_= "DetailsSummary--DetailsSummary--1DqhO DetailsSummary--fadeOnOpen--KnNyF")

#print(len(weather_data))

for day in weather_data:
    date = day.find('h3', class_="DetailsSummary--daypartName--kbngc").text
    #print(date)
    temp_section = day.find('div', class_="DetailsSummary--temperature--1kVVp")
    span_tags = temp_section.find_all('span')
    max_temp = span_tags[0].text
    min_temp = span_tags[2].span.text
    #print (max_temp)
    #print(min_temp)
    weather_condition = day.find('div', class_="DetailsSummary--condition--2JmHb").span.text
    #print(weather_condition)
    chance = day.find('div', class_="DetailsSummary--precip--1a98O").span.text
    #print(chance)
    wind_section = day.find('div', class_="DetailsSummary--wind--1tv7t DetailsSummary--extendedData--307Ax").span.text
    wind_separated = wind_section.split()
    wind_direction = wind_separated[0]
    wind_speed = wind_separated[1]
    #print(wind_direction)
    #print(wind_speed)
    final_data = (date, max_temp, min_temp, weather_condition, chance, wind_direction, wind_speed)
    print(final_data)
    with open('ELEC292_Lab2.txt', 'a', encoding='utf-8') as f:  #utf-8 Encoding used to degree symbol would procces properly
        print(final_data, file=f)
