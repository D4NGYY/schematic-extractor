import urllib.request
from pathlib import Path

downloads = [
    {
        "name": "arduino_uno_r3",
        "url": "https://content.arduino.cc/assets/UNO-TH_Rev3e_sch.pdf",
        "file": "arduino_uno_r3.pdf"
    },
    {
        "name": "esp32_devkitc",
        "url": "https://dl.espressif.com/dl/schematics/esp32_devkitc_v4-sch.pdf",
        "file": "esp32_devkitc.pdf"
    },
    {
        "name": "arduino_nano",
        "url": "https://content.arduino.cc/assets/NanoV3.3_sch.pdf",
        "file": "arduino_nano.pdf"
    }
]

def main():
    base_dir = Path("test_input/multi_schematic")
    base_dir.mkdir(parents=True, exist_ok=True)

    for item in downloads:
        dir_path = base_dir / item["name"]
        dir_path.mkdir(exist_ok=True)
        file_path = dir_path / item["file"]

        print(f"Downloading {item['name']} from {item['url']}...")
        try:
            req = urllib.request.Request(item["url"], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"  Saved to {file_path}")
        except Exception as e:
            print(f"  Error downloading {item['name']}: {e}")

if __name__ == "__main__":
    main()
