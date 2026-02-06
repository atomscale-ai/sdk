Quickstart
==========

Get up and running with the Atomscale SDK in under 5 minutes.

Install
-------

.. code-block:: bash

   pip install atomscale

Get Your API Key
----------------

1. Log in to the `Atomscale web app <https://app.atomscale.ai>`_
2. Go to **Profile > Account Management**
3. Copy your API key

Set it as an environment variable:

.. code-block:: bash

   export AS_API_KEY="your-api-key"

Or pass it directly in code (not recommended for production):

.. code-block:: python

   from atomscale import Client
   client = Client(api_key="your-api-key")

Create a Client
---------------

.. code-block:: python

   from atomscale import Client

   client = Client()

The client handles authentication automatically using the ``AS_API_KEY``
environment variable.

Upload Your First File
----------------------

.. code-block:: python

   client.upload(files=["my_rheed_video.mp4"])

Supported formats: ``.mp4``, ``.imm``, ``.png``, ``.jpg``, and XPS data files.

Find Your Data
--------------

.. code-block:: python

   results = client.search(keywords=["my_sample"])
   print(results[["Data ID", "Status", "Sample Name"]])

Access Analysis Results
-----------------------

.. code-block:: python

   analysed = client.get(results["Data ID"].to_list())

   # Get timeseries data (intensity, strain, etc.)
   df = analysed[0].timeseries_data
   print(df.columns)

Next Steps
----------

- :doc:`upload-files` - Upload RHEED videos, images, or XPS files
- :doc:`stream-rheed` - Stream live RHEED from your instrument
- :doc:`stream-timeseries` - Stream instrument sensor data (temperature, pressure)
- :doc:`search-and-download` - Find and download processed data
