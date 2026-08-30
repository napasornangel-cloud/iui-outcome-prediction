"""
Visits a Streamlit Community Cloud app with a real headless browser and,
if the app is showing the "Zzzz... this app has gone to sleep" screen,
clicks the wake-up button. A plain HTTP GET (requests/urllib) does NOT
work for this, because a sleeping app still returns HTTP 200 with a
static HTML shell -- the Streamlit process itself is not running, so
nothing short of an actual browser visit can trigger a restart.
"""

import os
import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set this to your app's URL, or override with the STREAMLIT_APP_URL
# environment variable (set as a GitHub Actions secret / env var so the
# URL isn't hardcoded if you ever change it).
APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://iui-outcome-prediction-memnxer5pxjwx2j6tapp2ry.streamlit.app/",
)

WAKE_BUTTON_XPATHS = [
    "//button[contains(., 'get this app back up')]",
    "//button[contains(., 'Yes, get this app back up')]",
    "//button[contains(text(), 'Wake')]",
]


def build_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,1696")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def main():
    driver = build_driver()
    try:
        print(f"Visiting {APP_URL} ...")
        driver.get(APP_URL)

        # Give the page a moment to render (sleeping-app screen is a
        # normal Streamlit component, not instant).
        time.sleep(5)

        wake_button = None
        for xpath in WAKE_BUTTON_XPATHS:
            buttons = driver.find_elements(By.XPATH, xpath)
            if buttons:
                wake_button = buttons[0]
                break

        if wake_button is not None:
            print("App was asleep. Clicking the wake-up button ...")
            wake_button.click()

            # Wait for the app to actually finish booting. We wait for
            # the sleeping-screen button to disappear as a simple signal
            # that Streamlit has started serving the real app.
            try:
                WebDriverWait(driver, 90).until(
                    EC.staleness_of(wake_button)
                )
                print("App appears to have woken up successfully.")
            except Exception:
                print(
                    "Warning: wake-up button did not disappear within "
                    "90 seconds. The app may still be booting; a manual "
                    "check is recommended."
                )
        else:
            print("App was already awake. Nothing to do.")

    except Exception as exc:
        print(f"ping_app.py failed: {exc}")
        sys.exit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
