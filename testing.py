import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import openai


import openai_functions as of
import utils
from bible import Bible
from db import BibleSummaryCollection


if __name__ == "__main__":
    openai.api_key = utils.load_api_key()
    bible = Bible("data/nrsv_bible.xml")

    print(bible[0][0][0].text)

    c = BibleSummaryCollection()