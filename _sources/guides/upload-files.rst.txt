Upload Files
============

Upload RHEED videos, images, or XPS files for analysis. Use this when you have
existing files on disk. For live instrument data, see :doc:`stream-rheed` or
:doc:`stream-timeseries`.

Supported Formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Extensions
   * - RHEED video
     - ``.mp4``, ``.avi``, ``.mov``, ``.webm``, ``.h5``, ``.imm``
   * - RHEED image
     - ``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.bmp``, ``.img``, ``.pdf``
   * - XPS
     - ``.vms``, ``.spe``, ``.spc``, ``.dat``, ``.txt``
   * - SEM
     - ``.tif``, ``.tiff``, ``.png``, ``.jpg``, ``.jpeg``, ``.bmp``
   * - Ellipsometry
     - ``.txt``, ``.csv``, ``.dat``
   * - Optical microscopy
     - ``.png``, ``.jpg``, ``.jpeg``, ``.tiff``, ``.tif``, ``.gif``, ``.bmp``, ``.img``, ``.pdf``
   * - Raman
     - ``.txt``, ``.csv``
   * - Photoluminescence
     - ``.txt``, ``.csv``
   * - SIMS
     - ``.xlsx``, ``.xls``
   * - OES
     - ``.txt``, ``.csv``
   * - Tool state/logs
     - ``.csv``, ``.txt``, ``.log``, ``.dat``, ``.xml``
   * - Archives
     - ``.zip``, ``.tar``, ``.gz``, ``.lzma``, ``.7z``, ``.rar``

Basic Upload
------------

.. code-block:: python

   from atomscale import Client

   client = Client()

   client.upload(files=[
       "/data/growths/2025-02-10/sample_001.mp4",
       "/data/growths/2025-02-10/sample_002.mp4",
   ])

You can mix different file types in a single upload call.

Upload with Sample Name
-----------------------

Link uploads to a sample for easier organization:

.. code-block:: python

   client.upload(
       files=["growth.mp4"],
       physical_sample="GaN-2025-001",
   )

Disable Progress Bars
---------------------

For scripts or CI environments:

.. code-block:: python

   client = Client(mute_bars=True)
   client.upload(files=["growth.mp4"])

What Happens Next
-----------------

1. Files upload to Atomscale (large files stream in chunks automatically)
2. Analysis pipelines start processing
3. Results appear in the web app and are searchable via the API

.. note::

   Processing runs asynchronously. Use :doc:`search-and-download` to check
   when analysis completes (``status="success"``).
