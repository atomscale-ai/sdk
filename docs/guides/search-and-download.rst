Search and Download
===================

Find data in the Atomscale catalogue and download processed results.

Basic Search
------------

.. code-block:: python

   from atomscale import Client

   client = Client()
   results = client.search(keywords=["GaN"])

   print(results[["Data ID", "Status", "Sample Name"]])

The search returns a DataFrame with matching entries from your organization.

Filter by Data Type
-------------------

.. code-block:: python

   # Only RHEED videos
   rotating = client.search(data_type="rheed_rotating")
   stationary = client.search(data_type="rheed_stationary")

   # Only images
   images = client.search(data_type="rheed_image")

   # Only XPS
   xps = client.search(data_type="xps")

Filter by Status
----------------

.. code-block:: python

   # Completed analysis
   completed = client.search(status="success")

   # Currently processing
   processing = client.search(status="running")

   # Active streams
   streaming = client.search(status="stream_active")

Available statuses: ``success``, ``pending``, ``running``, ``error``,
``stream_active``, ``stream_interrupted``, ``stream_finalizing``, ``stream_error``

Filter by Date or Duration
--------------------------

.. code-block:: python

   from datetime import datetime

   # Uploaded after Jan 1, 2025
   recent = client.search(
       upload_datetime=(datetime(2025, 1, 1), None),
   )

   # Growth longer than 30 minutes (1800 seconds)
   long_growths = client.search(
       growth_length=(1800, None),
   )

Combine Filters
---------------

.. code-block:: python

   results = client.search(
       keywords=["GaN"],
       data_type="rheed_stationary",
       status="success",
       growth_length=(1000, 5000),
   )

Search by ID
------------

.. code-block:: python

   exact = client.search(
       data_ids=["44fa63b0-74da-4d25-a362-2276c80a670a"]
   )

Limit to Personal Uploads
-------------------------

By default, search includes all data from your organization. To see only your
own uploads:

.. code-block:: python

   my_data = client.search(
       keywords=["demo"],
       include_organization_data=False,
   )

Download Processed Videos
-------------------------

.. code-block:: python

   client.download_videos(
       data_ids=results["Data ID"].to_list(),
       dest_dir="processed/",
   )

Files are saved as MP4 with the data ID as the filename.

.. caution::

   Processed videos can be large. Filter to only the IDs you need before
   downloading.
