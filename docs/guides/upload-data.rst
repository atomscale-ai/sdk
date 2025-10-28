Upload Data
===========

This guide adapts the *Uploading Data* section of ``general_use.ipynb``. Use it
to send new RHEED videos, images, or XPS files to Atomscale.

Choose files
------------

Collect the local file paths you want to upload. Mixing file types is fine.

.. code-block:: python

   files = [
       "/data/growths/2025-02-10/RHEED-stationary.mp4",
       "/data/growths/2025-02-10/RHEED-rotating.imm",
   ]

Start the upload
----------------

.. code-block:: python

   from atomicds.client import Client

   client = Client(api_key="YOUR_API_KEY")
   job = client.upload(files=files)

Each file streams to the API, and analysis starts as soon as data arrives. The
:meth:`upload` call returns a handle you can inspect for progress details.

Optional: mute progress bars
----------------------------

If you are running uploads non-interactively (for example in CI) pass
``mute_bars=True`` when constructing the client.

.. code-block:: python

   client = Client(api_key="YOUR_API_KEY", mute_bars=True)

Check status in the web app
---------------------------

Uploads immediately appear in the Atomscale UI. Analysis runs in the background,
and results land in the catalogue once the pipeline finishes.
