#!/usr/bin/env python3
from time import sleep
from os import mkdir
import wikipediaapi


def scrape():
    topics = [
        "FC_Bayern_München",
        "Borussia_Dortmund",
        "1._FC_Köln",
        "Eintracht_Frankfurt",
        "Borussia_Mönchengladbach",
        
        "VW",
        "BMW",
        "Automobile_Dacia",
        "Renault",
        "Škoda_Auto",

        "Bob_Marley",
        "Metallica",
        "Transsylvania_Phoenix",
        "Iris_(Romanian_band)",
        "Pink_Floyd"
    ]

    wiki = wikipediaapi.Wikipedia(
        user_agent="Students (students@uaic.ro)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    try:
        mkdir("./data")
    except:
        pass

    for topic in topics:
        page = wiki.page(topic)
        sleep(1)

        with open(f"./data/{topic}.txt", "w", encoding="utf-8") as file:
            file.write(page.text)


if __name__ == "__main__":
    scrape()
