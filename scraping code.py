import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

cities = ["Paris", "London", "New York", "Tokyo"]
CHECKIN_DATE = "2025-08-15"
CHECKOUT_DATE = "2025-08-20"
GROUP_ADULTS = 1
NO_ROOMS = 1

def get_hotels(city, max_results=500):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = (
        f"https://www.booking.com/searchresults.html"
        f"?ss={city}&checkin={CHECKIN_DATE}&checkout={CHECKOUT_DATE}"
        f"&group_adults={GROUP_ADULTS}&no_rooms={NO_ROOMS}"
        f"&selected_currency=USD"
    )

    hotels = []
    seen_ids = set()

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="property-card"]')))

        def collect_hotels():
            soup = BeautifulSoup(driver.page_source, "html.parser")
            hotel_cards = soup.find_all('div', {"data-testid": "property-card"})
            count_before = len(hotels)

            for hotel in hotel_cards:
                name = hotel.find("div", {"data-testid": "title"})
                name = name.text.strip() if name else "N/A"
                if name == "N/A":
                    continue

                card_id = name.lower()
                if card_id in seen_ids:
                    continue
                seen_ids.add(card_id)

                price = hotel.find("span", {"data-testid": "price-and-discounted-price"})
                price = price.text.strip() if price else "N/A"

                rating_container = hotel.find("div", {"data-testid": "review-score"})
                if rating_container:
                
                    score_div = rating_container.find("div", class_="f63b14ab7a dff2e52086")
                    rating = score_div.text.replace("Scored", "").strip() if score_div else "N/A"

                    category_div = rating_container.find("div", class_="f63b14ab7a f546354b44 becbee2f63")
                    rating_category = category_div.text.strip() if category_div else "N/A"

                    reviews_div = rating_container.find("div", class_="fff1944c52 fb14de7f14 eaa8455879")
                    num_reviews = reviews_div.text.strip() if reviews_div else "N/A"
                else:
                    rating = rating_category = num_reviews = "N/A"

                room_type_container = hotel.find("div", {"data-testid": "recommended-units"})
                room_type = room_type_container.find("h4") if room_type_container else None
                type_text = room_type.text.strip() if room_type else "N/A"


                distance = hotel.find("span", {"data-testid": "distance"})
                distance = distance.text.strip() if distance else "N/A"

                hotels.append({
                    'name': name,
                    'price': price,
                    'rating': rating,
                    'category': rating_category,
                    'number of reviews': num_reviews,
                    'type': type_text,
                    'Distance from Downtown': distance
                })

                if len(hotels) >= max_results:
                    break

            return len(hotels) > count_before

        while len(hotels) < max_results:
            if not collect_hotels():
                pass

            try:
                load_more = wait.until(EC.element_to_be_clickable((
                    By.XPATH, '//button[.//span[contains(text(), "Load more results")]]'
                )))
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more)
                time.sleep(1)
                load_more.click()
                time.sleep(4)
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="property-card"]')))
            except:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                if not collect_hotels():
                    break

        return hotels

    except Exception as e:
        print(f"Error scraping {city}: {e}")
        return []
    finally:
        driver.quit()

if __name__ == "__main__":
    all_hotels = []
    for city in cities:
        print(f"Fetching hotels for {city}...")
        hotels = get_hotels(city, max_results=500)
        print(f"Number of hotels found in {city}: {len(hotels)}")
        for hotel in hotels:
            print(hotel)
        all_hotels.extend(hotels)
    df = pd.DataFrame(all_hotels)
    df.to_csv("hotels_data.csv", index=False, encoding='utf-8-sig')