.. image:: images/orochi_icon.png
   :align: center
   :width: 80px
   :target: https://github.com/Tominaka-Laboratory/orochi.pixia
   :alt: Orochi Logo

============
orochi.pixia
============

.. |title| replace:: orochi.pixia
.. _title: https://Tominaka-Laboratory.github.io/orochi.pixia

.. |Icon| image:: images/orochi_icon.png
   :target: https://github.com/Tominaka-Laboratory/orochi.pixia
   :height: 100px

|PyPi| |Forge| |PythonVersion| |ChemRxiv| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |ChemRxiv| image:: https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15010802%2Fv2-blue.svg
        :target: https://doi.org/10.26434/chemrxiv.15010802/v2

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/Tominaka-Laboratory/orochi.pixia/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/Tominaka-Laboratory/orochi.pixia/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/Tominaka-Laboratory/orochi.pixia/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/Tominaka-Laboratory/orochi.pixia

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/orochi.pixia
        :target: https://anaconda.org/conda-forge/orochi.pixia

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/orochi.pixia
        :target: https://pypi.org/project/orochi.pixia/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/orochi.pixia
        :target: https://pypi.org/project/orochi.pixia/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/Tominaka-Laboratory/orochi.pixia/issues

Orochi PIXIA: 2D Area Detector Reduction & Geometry Calibration
================================================================

The ``orochi.pixia`` package provides advanced 2D area detector data reduction, instrumental geometry calibration, and azimuthal integration tailored for high-energy synchrotron total scattering. Operating on raw detector images (flat panels, TIFF, EDF), ``orochi.pixia`` extracts pristine 1D diffraction profiles *I(2θ)* and *I(Q)* free from oblique sensor penetration distortion and edge-truncation artifacts.

Key Features
------------
* **Detector Geometry Calibration**: Refines incident beam center (x₀, y₀), calibrated sample-to-detector distance L₀, and detector tilt/rotation angles against standard calibration materials (SRM CeO₂, Ni).
* **Physical Sensor Corrections**: Corrects for beam polarization, solid-angle distortion, detector thickness oblique absorption (μ_det · t_det), and capillary sample attenuation.
* **Automated Beamstop & Defect Masking**: Automatically identifies circular beamstop shadows, inter-module gaps, and dead pixels.
* **Diagonal Sector Integration**: Integrates over designated diagonal corner lobes (β = ±45° ± 18°, ±135° ± 18°), avoiding orthogonal detector edge boundaries to access maximum momentum transfer *Q_max* with invariant angular acceptance.

Quickstart: 2D Image Reduction
==============================

.. code-block:: python

    from orochi.pixia import Pixia
    import matplotlib.pyplot as plt

    # 1. Initialize Pixia
    pixia = Pixia()
    pixia.read_file("raw_detector_pattern.tif", target="sample")

    # 2. Configure calibrated beamline geometry
    pixia["beam_center_x"] = 1025.60
    pixia["beam_center_y"] = 1053.13
    pixia["distance"] = 227.575         # Sample-to-detector distance in mm
    pixia["rambda"] = 0.18033           # Wavelength in Angstrom
    pixia["pixel_size_x"] = 0.20        # Pixel pitch in mm
    pixia["pixel_size_y"] = 0.20

    # 3. Configure sector integration
    pixia["integration_mode"] = "sector" # 'full' (360 deg) or 'sector' (diagonal lobes)
    pixia["sector_angles"] = [-135, -45, 45, 135]
    pixia["sector_delta"] = 18.0

    # 4. Execute reduction pipeline
    pixia.run()

    # 5. Extract reduced 1D profile
    two_theta = pixia.tth
    intensity = pixia.intensity

    plt.plot(two_theta, intensity)
    plt.xlabel("2θ (deg)")
    plt.ylabel("Corrected Intensity (counts)")
    plt.show()

Citation
--------

If you use orochi.pixia in a scientific publication, please cite this package as:

    S. Tominaka, *Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials*, ChemRxiv (2026). DOI: 10.26434/chemrxiv.15010802/v2
    https://doi.org/10.26434/chemrxiv.15010802/v2

Installation
------------

The preferred method is to install using ``pip``:

.. code-block:: bash

    pip install orochi.pixia

To confirm that the installation was successful, type:

.. code-block:: bash

    python -c "import orochi.pixia; print('Orochi PIXIA imported successfully')"

If you prefer to install from source:

.. code-block:: bash

    git clone https://github.com/Tominaka-Laboratory/orochi.pixia.git
    cd orochi.pixia
    pip install -e .

Getting Started
---------------

You may consult our `online documentation <https://Tominaka-Laboratory.github.io/orochi.pixia>`_ for tutorials and API references.

Support and Contribute
----------------------

The orochi.pixia project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/Tominaka-Laboratory/orochi.pixia/issues>`_ and/or `submit a fix as a PR <https://github.com/Tominaka-Laboratory/orochi.pixia/pulls>`_.

Feel free to fork the project and contribute. To install orochi.pixia
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/Tominaka-Laboratory/orochi.pixia/blob/main/CODE_OF_CONDUCT.rst>`_.



License
-------

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE.rst` file for details.

This copyleft license guarantees that Orochi remains freely accessible, open, and transparent to the entire scientific community, and that all derivative works must likewise remain open source.

Citation
--------

If you use this package in academic research, please cite:

.. code-block:: bibtex

    @article{tominaka2026orochi,
      title   = {Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials},
      author  = {Tominaka, Satoshi},
      journal = {ChemRxiv},
      year    = {2026},
      doi     = {10.26434/chemrxiv.15010802/v2},
      url     = {https://doi.org/10.26434/chemrxiv.15010802/v2}
    }

Contact
-------

For more information on orochi.pixia please visit the project `web-page <https://Tominaka-Laboratory.github.io/>`_ or email Dr. Satoshi Tominaka at tominaka.satoshi@nims.go.jp.
