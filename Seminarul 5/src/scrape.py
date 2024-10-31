#!/usr/bin/env python3
import wikipediaapi


def scrape():
    wiki = wikipediaapi.Wikipedia(
        user_agent="Mihai (mihai.bojescu@uaic.ro)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )
    page = wiki.page("Romania")

    print(page.text)


if __name__ == "__main__":
    scrape()
