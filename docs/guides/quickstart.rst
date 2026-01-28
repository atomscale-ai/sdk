Quickstart
==========

This quickstart mirrors ``general_use.ipynb`` and walks through the essentials
for calling the Atomscale API.

Prerequisites
-------------

.. important::

   Before you begin, make sure you have:

   - Python 3.10 or newer
   - An active Atomscale account
   - An API key from the Atomscale web app (Profile → Account Management)

Install the client
------------------

.. code-block:: bash

   pip install atomscale

Create a client
---------------

The :class:`atomscale.client.Client` reads ``AS_API_KEY`` and
``AS_API_ENDPOINT`` from the environment. Export the variables, or pass values
explicitly if you prefer.

.. code-block:: python

   import os
   from atomscale.client import Client

   os.environ["AS_API_KEY"] = "YOUR_API_KEY"

   client = Client()

.. tip::

   Set ``AS_API_KEY`` in your shell profile (e.g., ``.bashrc`` or ``.zshrc``)
   to avoid hardcoding credentials in scripts.

Override the endpoint when pointing at staging or a private deployment:

.. code-block:: python

   client = Client(
       api_key="YOUR_API_KEY",
       endpoint="https://api.atomscale.ai/",
   )

.. warning::

   Never commit API keys to version control. Use environment variables or a
   secrets manager instead.

Next steps
----------

.. seealso::

   - :doc:`upload-data` – Upload RHEED videos, images, or XPS files
   - :doc:`search-data` – Find items in the catalogue
   - :doc:`inspect-results` – Explore results and plots
