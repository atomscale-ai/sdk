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

Filter by lifecycle state
-------------------------

``status`` accepts ``"success"``, ``"pending"``, ``"running"``, ``"error"``, and
the streaming-specific values ``"stream_active"``, ``"stream_interrupted"``,
``"stream_finalizing"``, and ``"stream_error"``.

.. code-block:: python

   completed = client.search(status="success")

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

Next steps
----------

Pass the ``Data ID`` column to :meth:`atomscale.client.Client.get` to fetch
analysis artefacts. See :doc:`inspect-results` for a hands-on tour.
