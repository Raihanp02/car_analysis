from playwright.sync_api import sync_playwright
import requests, os

url = "https://www.olx.co.id/item/tdp-cuma-38jt-mitsubishi-xpander-cross-15-rockford-edition-2021-at-iid-946453658"
os.makedirs("olx_images", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="networkidle")

    # Adjust selector based on actual gallery image tags
    img_urls = page.eval_on_selector_all(
        "img[src*='apollo.olx.co.id']",
        "elements => elements.map(e => e.src)"
    )
    img_urls = list(set(img_urls))  # dedupe

    for i, img_url in enumerate(img_urls):
        # request a higher-res version by tweaking the size param, e.g. ;s=1000x0
        high_res = img_url.split(";")[0] + ";s=1000x0"
        r = requests.get(high_res)
        if r.status_code == 200:
            with open(f"olx_images/img_{i}.jpg", "wb") as f:
                f.write(r.content)
            print(f"Saved img_{i}.jpg")

    browser.close()