import json

import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import openai


import openai_functions as of
import utils
from bible import Bible
from db import BibleSummaryCollection


if __name__ == "__main__":
    bible = Bible("data/nrsv_bible.xml", "data/chapter_index_map.json")


    print("Ruth".split(" "))
