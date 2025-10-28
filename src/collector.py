from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

def collect_posts(url, num_pages=1):
    driver = webdriver.Chrome()  # Requires chromedriver
    driver.get(url)
    posts = []
    
    for _ in range(num_pages):
        titles = driver.find_elements(By.CLASS_NAME, "post-title")  # Adjust selectors
        contents = driver.find_elements(By.CLASS_NAME, "post-content")
        dates = driver.find_elements(By.CLASS_NAME, "post-date")
        
        for t, c, d in zip(titles, contents, dates):
            posts.append({
                'title': t.text,
                'content': c.text,
                'date': d.text
            })
        
        next_button = driver.find_element(By.CLASS_NAME, "next-page")
        next_button.click()
        time.sleep(2)
    
    driver.quit()
    return pd.DataFrame(posts)

# Usage: df = collect_posts("https://game-forum.example.com")
