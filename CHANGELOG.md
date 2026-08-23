# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.2](https://github.com/RReverser/serde-ndim/compare/v2.2.1...v2.2.2) - 2026-08-23

### Other

- no-std fix and CI check ([#8](https://github.com/RReverser/serde-ndim/pull/8))

## [2.2.1](https://github.com/RReverser/serde-ndim/compare/v2.2.0...v2.2.1) - 2026-05-01

### Other

- Add CHANGELOG with historical releases
- Pass GITHUB_TOKEN to release-plz
- Use canonical release-plz/action name and enable verbose logging
- Automate releases via release-plz with OIDC trusted publishing ([#5](https://github.com/RReverser/serde-ndim/pull/5))
- Add support for bool arrays ([#4](https://github.com/RReverser/serde-ndim/pull/4))
- Fix docs.rs build ([#3](https://github.com/RReverser/serde-ndim/pull/3))

## [2.2.0] - 2025-12-28

### Changed
- Bumped `ndarray` to 0.17.

## [2.1.0] - 2025-08-17

### Added
- ndarray deserialization now accepts any `DataOwned` storage, matching `ndarray`'s own bound.

### Changed
- Bumped `nalgebra` to 0.34.

## [2.0.2] - 2024-08-19

### Fixed
- `no_std` build.

## [2.0.1] - 2024-08-19

### Fixed
- `Cargo.toml` dependency requirements that were not bumped together with the 2.0.0 code changes.

## [2.0.0] - 2024-08-19

### Changed
- Updated for `ndarray` 0.16 and `nalgebra` 0.33.

## [1.1.0] - 2023-04-28

### Added
- Support for non-standard memory layouts.

### Changed
- Simplified internal trait bounds.

## [1.0.0] - 2023-04-02

Initial release.

### Added
- Serde-based deserialization of n-dimensional arrays from self-describing formats (e.g. JSON).
- Optional integration with `ndarray`, `nalgebra`, and `arrayvec`.
- `no_std` support.
