# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- HPC (High Performance Computing) tools for parallel processing
  - LSF (bsub) job submission and monitoring
  - Batch processing capabilities
  - Progress tracking for job submission
  - Automatic resource management
- Import system updates including datetime and platform modules
- SMB tools for downloading Windows share files from Unix
- Interactive ROI selector with area removal/selection functionality
- S3 bucket upload and download functionality
- GIF creation functionality
- Parallel processing for image compression to parquet
- Seamless cloning methods
- Dataset checking functionality
- Multi-Otsu thresholding implementation
- Center crop functionality
- Image brightness adjustment
- Circle detection from single pin
- Split image with coordinate function
- Morphological operations

### Changed
- Updated imports for better system compatibility
- Improved interactive coordinate selection from images
- Enhanced core functionality with new image processing features
- Updated show_ function to support cmap with default gray
- Fixed read_img function for PIL image handling (gray=False)
- Optimized compress and filter functions with parallel processing
- Bug fixes in interactive ROI selector
- Updated core image processing functions

### Fixed
- Bug fix in read_config function
- Fixed conflicts in core notebook
- Corrected get_same_shape function
- Fixed overlay_mask_border_on_image function
- Resolved issues with RGB image reading
- Fixed Button import from ipywidgets

### Removed
- Removed TensorFlow dependencies for better environment compatibility
- Cleaned up notebook outputs using nbdev_clean

### Development
- Added pyarrow in requirements
- Updated project configuration in settings.ini
- Added documentation and examples
- Implemented sidebar for better navigation
- Added testing and validation
- Updated dependencies (matplotlib, scipy, skimage)
- Improved project structure and organization

## [1.0.0] - 2024-02-11
### Added
- Initial release with core CV tools functionality
- Basic image processing operations
- Core utilities for computer vision tasks

[Unreleased]: https://github.com/HasanGoni/cv_tools/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/HasanGoni/cv_tools/releases/tag/v1.0.0 