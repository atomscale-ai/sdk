Atomscale Python SDK
====================

|testing-badge| |tag-badge| |python-badge| |license-badge|

.. |testing-badge| image:: https://github.com/atomscale-ai/sdk/actions/workflows/testing.yml/badge.svg?branch=main
   :target: https://github.com/atomscale-ai/sdk/actions/workflows/testing.yml?query=branch%3Amain
   :alt: Testing status
.. |tag-badge| image:: https://img.shields.io/github/tag/atomscale-ai/sdk?include_prereleases=&sort=semver&color=blue
   :target: https://github.com/atomscale-ai/sdk/releases/
   :alt: Latest tag
.. |python-badge| image:: https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white
   :alt: Supported Python versions
.. |license-badge| image:: https://img.shields.io/badge/License-MPL_2.0-blue
   :target: https://github.com/atomscale-ai/sdk/blob/main/LICENSE
   :alt: License: MPL 2.0

The ``atomscale`` package connects your Python code to the Atomscale platform
for RHEED analysis, XPS processing, and instrument data management.

Installation
------------

.. code-block:: bash

   pip install atomscale

Set your API key (from Profile > Account Management in the web app):

.. code-block:: bash

   export AS_API_KEY="your-api-key"

Quick Example
-------------

.. code-block:: python

   from atomscale import Client

   client = Client()

   # Upload a RHEED video for analysis
   client.upload(files=["growth_001.mp4"])

   # Find your data
   results = client.search(keywords=["GaN"])

   # Get analysis results
   analysed = client.get(results["Data ID"].to_list())
   print(analysed[0].timeseries_data)

What Can You Do?
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Send data**
     - Upload files or stream live RHEED/instrument data to Atomscale
   * - **Get results**
     - Search the catalogue, download processed videos, access analysis data
   * - **Monitor live**
     - Poll for real-time updates during active streaming sessions

Guides
------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   guides/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Send Data to Atomscale

   guides/upload-files
   guides/stream-rheed
   guides/stream-timeseries

.. toctree::
   :maxdepth: 1
   :caption: Retrieve Results

   guides/search-and-download
   guides/analysis-results
   guides/monitor-live

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Support
-------

- **Issues**: `GitHub <https://github.com/atomscale-ai/sdk>`_
- **Email**: support@atomscale.ai
