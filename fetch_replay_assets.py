"""Fetch only the real replay assets via HTTP ranges in pinned upstream ZIPs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import posixpath
import struct
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib

import requests
import yaml

BASE = "https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/"
ROOT = Path(__file__).resolve().parent / "RoboTwin"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


class RemoteZip(io.RawIOBase):
    def __init__(self, name):
        self.url = BASE + name
        response = session().get(self.url, headers={"Range": "bytes=-65536"}, timeout=25)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Server did not honor bounded range request")
        content_range = response.headers["Content-Range"]
        self.size = int(content_range.split("/")[1])
        self.tail = response.content
        self.tail_start = self.size - len(self.tail)
        self.position = 0
        self.zip = zipfile.ZipFile(self)
        self.members = {info.filename: info for info in self.zip.infolist()}

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        data = self.range(self.position, min(self.position + count, self.size))
        self.position += len(data)
        return data

    def range(self, start, end):
        if start >= self.tail_start:
            return self.tail[start - self.tail_start:end - self.tail_start]
        response = session().get(self.url, headers={"Range": "bytes=%d-%d" % (start, end - 1)}, timeout=60)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - start:
            raise RuntimeError("Invalid bounded ZIP range response")
        return response.content

    def find(self, suffix):
        matches = [name for name in self.members if name == suffix or name.endswith("/" + suffix)]
        if len(matches) != 1:
            raise RuntimeError("Ambiguous/missing upstream archive member: " + suffix)
        return matches[0]

    def extract(self, name, destination):
        info = self.members[name]
        # Fetch compressed member plus its small local header in one HTTP request.
        end = min(info.header_offset + info.compress_size + 4096, self.size)
        block = self.range(info.header_offset, end)
        header = struct.unpack("<IHHHHHIIIHH", block[:30])
        if header[0] != 0x04034B50:
            raise RuntimeError("Invalid ZIP local header")
        offset = 30 + header[-2] + header[-1]
        if offset + info.compress_size > len(block):
            block = self.range(info.header_offset, info.header_offset + offset + info.compress_size)
        compressed = block[offset:offset + info.compress_size]
        data = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
        if len(data) != info.file_size or zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
            raise RuntimeError("ZIP member size/CRC mismatch: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return {"path": str(destination.relative_to(ROOT)), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "archive_member": name,
                "compressed_bytes": info.compress_size, "source": self.url}


def main():
    began = time.monotonic()
    robot_archive = RemoteZip("embodiments.zip")
    prefix = robot_archive.find("aloha-agilex/config.yml").rsplit("/", 1)[0]
    robot_root = ROOT / "assets/embodiments/aloha-agilex"
    records = [robot_archive.extract(prefix + "/config.yml", robot_root / "config.yml")]
    config = yaml.safe_load((robot_root / "config.yml").read_text())
    urdf_relative = posixpath.normpath(config["urdf_path"])
    urdf_member = posixpath.normpath(prefix + "/" + urdf_relative)
    urdf = robot_root / urdf_relative
    records.append(robot_archive.extract(urdf_member, urdf))
    needed = set()
    for element in ET.parse(urdf).iter():
        filename = element.get("filename")
        if filename:
            member = posixpath.normpath(posixpath.dirname(urdf_member) + "/" + filename.removeprefix("file://").removeprefix("package://"))
            if member not in robot_archive.members:
                raise FileNotFoundError("URDF referenced member not in ZIP: " + member)
            needed.add(member)

    def robot_member(member):
        relative = posixpath.relpath(member, prefix)
        if relative.startswith("../"):
            raise RuntimeError("Robot dependency escaped asset directory")
        return robot_archive.extract(member, robot_root / relative)

    with ThreadPoolExecutor(max_workers=4) as executor:
        records.extend(executor.map(robot_member, sorted(needed)))
    textures = set()
    for record in records:
        path = ROOT / record["path"]
        if path.suffix.lower() != ".dae":
            continue
        for image in ET.parse(path).iter():
            if image.tag.rsplit("}", 1)[-1] == "image":
                for element in image.iter():
                    if element.tag.rsplit("}", 1)[-1] == "init_from" and element.text:
                        member = posixpath.normpath(posixpath.dirname(record["archive_member"]) + "/" + element.text.strip().removeprefix("file://"))
                        textures.add(member)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records.extend(executor.map(robot_member, sorted(textures - needed)))
    print(json.dumps({"robot_files": len(records), "robot_raw_bytes": sum(r["bytes"] for r in records),
                      "robot_compressed_bytes": sum(r["compressed_bytes"] for r in records)}), flush=True)
    objects = RemoteZip("objects.zip")
    targets = []
    for model in (13, 16):
        for suffix in ("visual/base%d.glb" % model, "collision/base%d.glb" % model, "model_data%d.json" % model):
            member = objects.find("001_bottle/" + suffix)
            targets.append((member, ROOT / "assets/objects/001_bottle" / suffix))
    with ThreadPoolExecutor(max_workers=4) as executor:
        records.extend(executor.map(lambda item: objects.extract(*item), targets))
    source_root = "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/c3ddfa8b97d5519efa828b075999bd0006778e5e/"
    for relative in ("task_config/demo_clean.yml", "task_config/_camera_config.yml"):
        response = session().get(source_root + relative, timeout=20)
        response.raise_for_status()
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        records.append({"path": relative, "bytes": len(response.content), "source": source_root + relative,
                        "sha256": hashlib.sha256(response.content).hexdigest(), "compressed_bytes": len(response.content)})
    result = {"source": "Pinned official RoboTwin asset archives; bounded HTTP ZIP member reads",
              "files": records, "elapsed_s": time.monotonic() - began,
              "raw_bytes": sum(r["bytes"] for r in records),
              "compressed_member_bytes": sum(r["compressed_bytes"] for r in records)}
    (ROOT.parent / "replay-assets-manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
