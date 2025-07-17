# Multi Spectral Camera

User Interface to control Camera and a device attached to the Serial Port.
Blackfly as well as OpenCV supported cameras.
Basler Cameras will be added once USB and Blackfly work.

A Lightsource is typically connected to the serial interface. For example https://github.com/uutzinger/blackflysync/

Urs Utzinger
2022, 2023, 2025

##  Installation

### Dependencies

- Python
    - `pyqt5` or `pyqt6` user interface
    - `pyqtgraph` display
    - `numpy` data gathering and manipulation
    - `markdown` help file
    - `scipy` image decompression
    - `fastplotlib` for high throughput plotting
    - `numba` acceleration of numpy code

- Camera Support
    - USB and internal Camera Support `pip3 install opencv-contrib-python`
    - [FLIR Spinnaker Support](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK)
    - [Basler Camera Support](https://www.baslerweb.com/en-us/software/pylon/)
