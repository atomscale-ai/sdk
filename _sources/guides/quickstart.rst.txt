Quickstart
==========

This quickstart mirrors ``general_use.ipynb`` and walks through the essentials
for calling the Atomscale API.

Prerequisites
-------------

- Python 3.10 or newer
- An active Atomscale account
- An API key from the Atomscale web app (Profile → Account Management)

Install the client
------------------

.. code-block:: bash

   pip install atomicds

Create a client
---------------

The :class:`atomicds.client.Client` reads ``AS_API_KEY`` and
``AS_API_ENDPOINT`` from the environment. Export the variables, or pass values
explicitly if you prefer.

.. code-block:: python

   import os
   from atomicds.client import Client

   os.environ["AS_API_KEY"] = "YOUR_API_KEY"

   client = Client()

Override the endpoint when pointing at staging or a private deployment.

.. code-block:: python

   client = Client(
       api_key="YOUR_API_KEY",
       endpoint="https://api.atomscale.ai/",
   )

Next steps
----------

- Upload files with :doc:`upload-data`
- Find items in the catalogue with :doc:`search-data`
- Explore results and plots with :doc:`inspect-results`
