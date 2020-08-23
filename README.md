<!--

-->





<!-- PROJECT SHIELDS -->
<!--

-->
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/azzeddinetiba/FeCLAP">
    <img src="logo/logo_logo.png" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">FeCLAP</h3>

  <p align="center">
    A simple solver for laminate composite plates
    <br />
    <a href="https://github.com/azzeddinetiba/FeCLAP"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/azzeddinetiba/FeCLAP">View Demo</a>
    ·
    <a href="https://github.com/azzeddinetiba/FeCLAP/issues">FeCLAP Bug</a>
    ·
    <a href="https://github.com/azzeddinetiba/FeCLAP/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]]()

This is a personal project small solver for the **F**inite **e**lement analysis of **C**omposite **L**aminate **P**lates = **_FeCLAP_** for rectangular geometries.
This solver supports **static, modal, transient and non linear analysis** using a perfectly elasto-plastic model.


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.
Further explanations about user inputs and results will be available soon.

### Prerequisites

This project requires the following libraries:
* Numpy
```sh
pip install numpy
```

* Scipy
```sh
pip install scipy
```

* Cython
```sh
pip install cython
```

* Eigency
```sh
pip install eigency
```

### Installation
 
* Clone the FeCLAP
```sh
git clone https://github.com/azzeddinetiba/FeCLAP.git
```

* Install the required packages

* Delete _FeCLAP\NonLinearModule\NonLinearModule.cpp_

* Build the NonLinearModule (this build uses Cython and C++)
```sh
python setup.py build
python setup.py install
```
* Add to _FeCLAP\build\lib.win32-3.7\NonLinearModule\__init__.py_
the line :
```sh
from .NonLinearModule import * 
```

* Copy the  _\lib.win32-3.7\NonLinearModule_ files to the
_FeCLAP\NonLinearModule_ folder

<!-- USAGE EXAMPLES -->
## Usage

(Under construction)

_For more examples, please refer to the [Documentation]( ) (Under construction)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/azzeddinetiba/FeCLAP/issues) for a list of proposed features (and known issues).



<!-- CONTACT -->
## Contact

TIBA Azzeddine - [Portfolio](https://portfolium.com/AzzeddineTiba/portfolio) - azzeddine.tiba@gmail.com

Project Link: [FeCLAP Github Link](https://github.com/azzeddinetiba/FeCLAP)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [P.-O. Persson, G. Strang, A Simple Mesh Generator in MATLAB. SIAM Review, Volume 46 (2), pp. 329-345, June 2004](https://github.com/bfroehle/pydistmesh)
* [KRISTIAN KRABBENHØFT, BASIC COMPUTATIONAL PLASTICITY, June 2002](http://homes.civil.aau.dk/lda/continuum/)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/azzeddinetiba/FeCLAP/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/azzeddinetiba/FeCLAP/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/azzeddinetiba/FeCLAP/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/azzeddine-tiba/
[product-screenshot]: logo/hoffman.png
