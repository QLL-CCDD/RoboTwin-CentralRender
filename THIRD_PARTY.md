# Source provenance and licenses

This repository is an experiment handoff, not a relicensing of upstream projects.
Original notices in vendored files are retained.

- `native-src/` and `references/`: RoboTwin-Platform/RoboTwin, pinned
  `c3ddfa8b97d5519efa828b075999bd0006778e5e`.
  See `licenses/RoboTwin-LICENSE` and `native-source-manifest.json`.
- SDK wheel 0.8.0b31, launcher, campaign tools, Policy adapter files and frozen
  manifests originate from QLL-CCDD/RoboScope-Kit. See
  `licenses/RoboScope-Kit-LICENSE`, bundle manifests and provenance hashes.
- The external StarVLA runtime is pinned to
  `7e4a47360093d79dc42b883885c04fbe790c5b60`; its code and model weights are not
  bundled here. `licenses/StarVLA-LICENSE` records the source license; model
  terms are supplied by the upstream model repository and are separate.
- RoboTwin assets, SAPIEN, CUDA, Torch and the complete runtime environments
  are not bundled. Fetch scripts/manifests identify upstream sources.

The original stage-3 and closed-loop manifests record their historical archives.
`EXPORT_MANIFEST.json` hashes this merged handoff snapshot, excluding itself.
