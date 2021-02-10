import requests
import re
from html.parser import HTMLParser
parser = HTMLParser()

final_list = []
for (part, num_sites) in [("maennlich", 13), ("weiblich", 14)]:
    for site in range(num_sites):
        r = requests.get(f"https://www.vornamen-weltweit.de/{part}-deutsch.php?Seite={str(site)}")
        www = r.text
        l = re.findall("[0-9]\">.*?</a></div>\n\t\t\t", www)
        l = [x.split(">")[1].split("<")[0] for x in l]
        final_list.extend(l)

final_list = [parser.unescape(s) for s in final_list]
with open("./datasets/german_names.txt", 'w') as f:
    f.write("\n".join(final_list))