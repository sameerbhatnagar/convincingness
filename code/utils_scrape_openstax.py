import os
import codecs
import json
import plac
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

import data_loaders

OPENSTAX_TEXTBOOK_DISCIPLINES = {
    "Chemistry": ["chemistry-2e"],
    "Biology": ["biology-2e"],
    "Physics": [
        "university-physics-volume-3",
        "university-physics-volume-2",
        "university-physics-volume-1",
    ],
    "Statistics": ["introductory-statistics"],
    "Ethics":["business-ethics","video-text-clean"],
}


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers["Content-Type"].lower()
    return (
        resp.status_code == 200
        and content_type is not None
        and content_type.find("html") > -1
    )


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def write_text_to_file(l, textbook_dir, base_url):
    text = None
    fname = os.path.join(textbook_dir, l + ".txt")

    if l[2:] == "key-terms":
        text = {
            k.text: v.text
            for (k, v) in zip(
                BeautifulSoup(
                    simple_get(base_url + l), "html.parser"
                ).find_all("dt"),
                BeautifulSoup(
                    simple_get(base_url + l), "html.parser"
                ).find_all("dd"),
            )
        }
        with open(fname, "w") as f:
            json.dump(text, f, indent=2)
        return

    elif l[2:] == "summary":
        pass
    #     for h3 in BeautifulSoup(simple_get(base_url+l),"html.parser").
    # find_all("h3"):
    #         if h3.text!="Summary":
    #             print h3.text
    #             for h4 in h3.next_sibling():
    #                 print
    # #                 print([li.text for li in list(h4.children)
    # if li.name=="li"])
    elif l[2:] == "introduction":
        text = [
            t.text
            for t in BeautifulSoup(
                simple_get(base_url + l), "html.parser"
            ).find_all("p")[:-2]
        ]
    else:
        try:
            if type(int(l.split("-")[0])) == int:
                text = [
                    t.text
                    for t in BeautifulSoup(
                        simple_get(base_url + l), "html.parser"
                    ).find_all("p")[:-2]
                ]
        except ValueError as e:
            print("\t skipping " + l)
            text = None

    if text:
        print("\t -" + l)
        with codecs.open(fname, "w", encoding="utf-8") as f:
            for t in text:
                f.write(t)

    return


def scrape_textbooks(discipline: (
    "Discipline",
    "positional",
    None,
    str,
    ["Physics", "Biology", "Chemistry", "Ethics"],
),):
    print(discipline)
    textbook_names=OPENSTAX_TEXTBOOK_DISCIPLINES[discipline]

    for textbook_name in textbook_names:
        textbook_dir = os.path.join(
            data_loaders.BASE_DIR,
            os.pardir,
            "textbooks",
            discipline,
            textbook_name,
        )
        if not os.path.exists(textbook_dir):
            os.makedirs(textbook_dir)
        print(textbook_name)

        intro_url = (
            "https://openstax.org/books/"
            + textbook_name
            + "/pages/1-introduction"
        )
        base_url = intro_url[:-14]
        raw_html = simple_get(url=intro_url)
        html = BeautifulSoup(raw_html, "html.parser")
        links = [l.get("href") for l in html.select("a")][7:]
        for l in links:
            if not os.path.exists(os.path.join(textbook_dir, l + ".txt")):
                write_text_to_file(l, textbook_dir, base_url)


    return


if __name__ == "__main__":

    plac.call(scrape_textbooks)
