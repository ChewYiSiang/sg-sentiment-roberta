import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

url = "https://forums.hardwarezone.com.sg/forums/eat-drink-man-woman.16/page-1"
response = requests.get(url, headers=HEADERS, timeout=10)

soup = BeautifulSoup(response.text, "html.parser")

# Print all unique div classes to find the right one
all_classes = set()
for div in soup.find_all("div", class_=True):
    for c in div.get("class", []):
        all_classes.add(c)

print("\nAll div classes found:")
for c in sorted(all_classes):
    print(" ", c)