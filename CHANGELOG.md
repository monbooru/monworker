# Changelog

## [v1.0.2] - 2026-11-20
### Fixed
- monbooru's Stats shows the models a worker has loaded during its first job after a start, instead of `not loaded`.
- A worker removed in monbooru while it was down offers itself again at its next start, and one moved to another port or machine is found at its new address.
- An offer monbooru refuses, such as a name another worker already has, stops with monbooru's reason in the log instead of being made again and again.
- The models folder is created at start when it is missing.
- The start warning no longer says no model will run when ONNX Runtime is not beside the worker but may still be on the system's library path.

## [v1.0.1] - 2026-11-18
### Changed
- The images are tagged with the version's `v`, like monbooru's and monloader's: `v1.0.1`, `v1.0.1-cuda`, `v1.0.1-openvino`.

## [v1.0.0] - 2026-11-18
### Added
- Initial release.
