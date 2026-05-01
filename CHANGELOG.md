# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
