from bs4 import BeautifulSoup
import urllib2
import pandas as pd
import sys

def scrape(site, lang_name):
        filename = "data/terms_1_to_100/" + lang_name + ".csv"
        page = urllib2.urlopen(site)
        table = pd.io.html.read_html(page, encoding="utf8")[0]

        table.to_csv(filename, index=False, header=False, encoding="utf8") 

if __name__ == "__main__":
        if len(sys.argv) != 2:
                print("Usage: python scrape_num.py language")
        lang = sys.argv[1]
        addr = "http://www.sf.airnet.ne.jp/ts/language/number/" + lang + ".html"
	scrape(addr, lang)
