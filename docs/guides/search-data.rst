Search the Catalogue
====================

Use :meth:`atomscale.client.Client.search` to locate uploaded data. The examples
below mirror ``general_use.ipynb`` and add a few extra filters you can combine.

Basic search
------------

.. code-block:: python

   from atomscale.client import Client

   client = Client(api_key="YOUR_API_KEY")

   rheed_runs = client.search(keywords=["WSe2"])
   print(rheed_runs[["Data ID", "Status", "Sample Name"]])

.. tip::

   Keywords are matched against sample names, file names, and metadata fields.
   Use specific terms for more precise results.

Limit to your uploads
---------------------

.. code-block:: python

   personal_only = client.search(
       keywords="demo",
       include_organization_data=False,
   )

Filter by IDs or type
---------------------

.. code-block:: python

   exact = client.search(data_ids=["44fa63b0-74da-4d25-a362-2276c80a670a"])

   rotating = client.search(data_type="rheed_rotating")

.. list-table:: Available data types
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Description
   * - ``rheed_stationary``
     - Stationary RHEED video
   * - ``rheed_rotating``
     - Rotating RHEED video
   * - ``rheed_image``
     - Single RHEED image
   * - ``xps``
     - XPS spectrum data

Filter by lifecycle state
-------------------------

``status`` accepts ``"success"``, ``"pending"``, ``"running"``, ``"error"``, and
the streaming-specific values ``"stream_active"``, ``"stream_interrupted"``,
``"stream_finalizing"``, and ``"stream_error"``.

.. code-block:: python

   completed = client.search(status="success")

.. note::

   Use ``status="stream_active"`` to find live streaming sessions that are
   currently receiving data.

Apply numeric or datetime bounds
--------------------------------

You can pass ``(min, max)`` tuples for growth length (seconds),
upload timestamp, or last-accessed timestamp. Use ``None`` for an open bound.

.. code-block:: python

   from datetime import datetime

   recent = client.search(
       upload_datetime=(datetime(2025, 1, 1), None),
       growth_length=(3000, None),
   )

.. tip::

   Combine multiple filters in a single call for precise queries:

   .. code-block:: python

      results = client.search(
          keywords="GaN",
          data_type="rheed_stationary",
          status="success",
          growth_length=(1000, 5000),
      )

Next steps
----------

Pass the ``Data ID`` column to :meth:`atomscale.client.Client.get` to fetch
analysis artefacts.

.. seealso::

   :doc:`inspect-results` – Work with timeseries, diffraction graphs, and videos
