import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# scraping hardwarezone forums for local Singaporean sentiment data
# it contains forum discussions on food, relationships, and daily life. This produces highly opinionated, emotionally expressive posts in natural Singlish.
# a tech subforum would be more neutral and factual, which is less useful for sentiment training.

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

BASE_FORUM_URL = "https://forums.hardwarezone.com.sg/forums/eat-drink-man-woman.16/page-{}"

# 2 level scraping: level 1 = get thread URLs from listing pages, level 2 = visit each thread and scrape posts inside
# === Level 1: Get thread URLs from the listing pages ===
def get_thread_urls(page_num: int) -> list:
    url = BASE_FORUM_URL.format(page_num)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        urls = []
        # structItem--thread contains each thread row in the listing
        for item in soup.find_all("div", class_="structItem--thread"):
            title_div = item.find("div", class_="structItem-title")
            if title_div:
                link = title_div.find("a", attrs={"data-tp-primary": True})
                if not link:
                    # fallback: grab any anchor inside the title div
                    link = title_div.find("a", href=True)
                if link and link.get("href"):
                    full_url = "https://forums.hardwarezone.com.sg" + link["href"]
                    urls.append(full_url)

        return urls

    except requests.exceptions.RequestException as e:
        print(f"  Failed to get thread list page {page_num}: {e}")
        return []


# === Level 2: Scrape posts inside a thread ===
def scrape_thread(url: str) -> list:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        posts = []
        for post in soup.find_all("div", class_="bbWrapper"):
            text = post.get_text(separator=" ", strip=True)
            # 40 chars as lower bound to filter out lacking sentiments. 1000 chars as upper bound to filter out extremely long posts which may contain mixed sentiments or off-topic discussions
            if 40 < len(text) < 1000:
                posts.append({
                    "text": text,
                    "source": "hardwarezone",
                    "label": None
                })

        return posts

    except requests.exceptions.RequestException as e:
        print(f"  Failed to scrape thread {url}: {e}")
        return []


# === Main scraping loop ===
all_posts = []
all_thread_urls = []

# Step 1: collect thread URLs from 10 listing pages
print("=== Collecting thread URLs ===")
for page_num in range(1, 11):  # 10 listing pages ~ 200-250 thread URLs
    print(f"Listing page {page_num}/10...")
    urls = get_thread_urls(page_num)
    all_thread_urls.extend(urls)
    print(f"  Found {len(urls)} threads | Total so far: {len(all_thread_urls)}")
    time.sleep(2)

print(f"\nTotal threads to scrape: {len(all_thread_urls)}")

# Step 2: visit each thread and collect posts
print("\n=== Scraping thread content ===")
for i, url in enumerate(all_thread_urls):
    if i % 20 == 0:
        print(f"Thread {i}/{len(all_thread_urls)} | Posts collected: {len(all_posts)}")
    posts = scrape_thread(url)
    all_posts.extend(posts)
    time.sleep(2)  # polite delay between thread requests

# === Save ===
df = pd.DataFrame(all_posts).drop_duplicates(subset="text")
df.to_csv("data/raw_hwz.csv", index=False)

print(f"\n=== Done ===")
print(f"Total unique posts collected: {len(df)}")
print("\nSample posts:")
for text in df["text"].head(3):
    print(f"  - {text[:100]}...")