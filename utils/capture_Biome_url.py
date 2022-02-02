# coding = utf-8
import os
import requests
import re

url = "https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data"

res = requests.get(url)
res.encoding = 'utf-8'


# TODO: 202 / 206
res = res.text
print(res)
# it = re.finditer(r'''<a href="https://(.*).(.*).(.*)/(.*)/cca_irish_2015/(.*).(.*).(.*)">''', res)
# it = re.finditer(r"https://(.*).(.*).(.*)/(.*)/cca_irish_2015/(.*).(.*).gz", res)
it = re.finditer(r"htt(.*)://landsat.usgs.gov/cloud-validation/cca_irish_2015/(.*).tar.gz", res)
with open("urls_l7_https.txt", "w+") as f:
    for match in it:
        # print (match.group() )
        f.write(match.group() + '\n')
