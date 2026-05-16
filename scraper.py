import json
import re
import requests

URL = (
    "https://tcp-us-prod-rnd.shl.com/"
    "voiceRater/shl-ai-hiring/"
    "shl_product_catalog.json"
)


def clean_json_text(text):

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        text,
    )

    return text


def main():

    print("Downloading SHL catalog...")

    response = requests.get(URL, timeout=30)

    response.raise_for_status()

    raw_text = response.text

    cleaned_text = clean_json_text(raw_text)

    try:

        data = json.loads(
            cleaned_text,
            strict=False
        )

    except Exception as e:

        print(f"\nJSON parsing failed:\n{e}")

        with open(
            "raw_catalog.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(cleaned_text)

        return

    with open(
        "catalog.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved {len(data)} items "
        f"to catalog.json"
    )

    print("\nSample entries:\n")

    for item in data[:5]:

        print("=" * 50)

        print(item.get("name", "NO NAME"))

        print(item.get("link", "NO URL"))


if __name__ == "__main__":
    main()