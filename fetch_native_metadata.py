"""Fetch only the small native import metadata and aloha planner SRDF."""
import json

from fetch_replay_assets import RemoteZip, ROOT


def main():
    objects = RemoteZip("objects.zip")
    records = [objects.extract(objects.find(name), ROOT / "assets/objects" / name)
               for name in ("objaverse/list.json", "same.json")]
    robots = RemoteZip("embodiments.zip")
    name = "aloha-agilex/srdf/arx5_description_isaac.srdf"
    records.append(robots.extract(robots.find(name), ROOT / "assets/embodiments" / name))
    output = ROOT.parent / "native-extra-assets-manifest.json"
    output.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
