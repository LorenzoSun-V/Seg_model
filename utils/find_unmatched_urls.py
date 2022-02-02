import os


unmatched_url = []
with open("urls_l7.txt", encoding="utf-8") as f1:
    lines_206 = f1.readlines()
    with open("urls_l7_https.txt", encoding="utf-8") as f2:
        lines_202 = f2.readlines()
        for line in lines_206:
            new_line = "https://" + line
            if new_line not in lines_202:
                unmatched_url.append(new_line)

print(unmatched_url)

