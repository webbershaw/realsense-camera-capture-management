# Realsense Camera Capture Management

The **Realsense Camera Capture Management** software facilitates real-time image capture and saving from Realsense cameras using PyRealsense, OpenCV, and PIL. It supports multiple cameras and offers two main implementation methods with distinct functionalities and interfaces.

## Features

- **Real-time image capture** using Realsense cameras.
- Supports **multiple cameras** simultaneously.
- Image saving functionality using OpenCV and PIL libraries.
- Two distinct implementation methods:
  - Web-based capture feature using Django (outdated and no longer maintained).
  - Desktop-based GUI application in Python.

## Implementation Methods

### ~~Method 1: Django-based Web Application (Deprecated)~~
- ~~This method is based on Django and allows users to capture images via a web interface.~~
- ~~**Limitations:**~~
  - ~~Does not support automatic alignment of depth cameras with corresponding RGB cameras.~~
  - ~~Occasional freezing bugs.~~
  - ~~This version is outdated and no longer maintained.~~

### Method 2: Python-based GUI Application
- A standalone desktop application developed in Python.
- **File Path:** `PythonUI/`
- **To Run the Application:**
  ```bash
  sudo python3 ./PythonUI/start.py
  ```

#### Controls:
- **q**: Quit the application.
- **b**: Decrease the index of the current image by 1.
- **n**: Decrease the index of the current image set by 1.
- **p**: Start the next set of image captures.
- **s**: Capture and save images from all cameras at the current moment.

## Upcoming Features
- Integration of UI buttons to complement keyboard operations, enhancing user interaction and accessibility.

## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/webbershaw/realsense-camera-capture-management.git
```

Navigate to the project directory:
```bash
cd realsense-camera-capture-management
```

Create a virtual environment:
```bash
python3 -m venv venv
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Follow the specific run commands listed under each implementation method. Ensure you have the necessary permissions and environment settings to execute the application.

## Thanks
- @lcandy2

## Contributing
Contributions are welcome, especially in areas such as improving existing functionalities and extending the software to include new features. Please submit a pull request or open an issue to discuss proposed changes or additions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

