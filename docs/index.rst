atomscale API Client
===================

.. |testing-badge| image:: https://github.com/atomscale-ai/sdk/workflows/Testing/badge.svg
   :target: https://github.com/atomscale-ai/sdk/actions?query=workflow%3A%22Testing%22
   :alt: Testing status
.. |tag-badge| image:: https://img.shields.io/github/tag/atomscale-ai/sdk?include_prereleases=&sort=semver&color=blue
   :target: https://github.com/atomscale-ai/sdk/releases/
   :alt: Latest tag
.. |python-badge| image:: https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white
   :alt: Supported Python versions
.. |license-badge| image:: https://img.shields.io/badge/License-GPLv3-blue
   :target: https://github.com/atomscale-ai/sdk/blob/main/LICENSE
   :alt: License: GPLv3

|testing-badge| |tag-badge| |python-badge| |license-badge|

``atomscale`` is the Python SDK for the Atomscale platform. Use these docs to
install the client, move data, and integrate streaming or polling workflows.
The material is split into:

* Task-focused guides that mirror the published notebooks.
* Auto-generated API references pulled directly from the source.

If you need help from a human, open an issue on
`GitHub <https://github.com/atomscale-ai/sdk>`_ or email
support@atomscale.ai.

Features
--------

- Unified :class:`atomscale.client.Client` for uploads, catalogue search, and downloads.
- Streaming helpers for live RHEED capture via push or generator-style interfaces.
- Fine-grained search filters for IDs, data types, lifecycle states, and time bounds.
- Rich result objects exposing timeseries, diffraction graphs, and processed videos.
- Polling utilities for synchronous, threaded, or async consumption of updates.

Installation
------------

.. code-block:: bash

  pip install atomscale

Set the ``AS_API_KEY`` and (optional) ``AS_API_ENDPOINT`` environment variables
before creating a :class:`~atomscale.client.Client`, or pass them directly when
constructing the client in your scripts.

.. note::

   The package was renamed from ``atomicds``. Importing ``atomicds`` still
   works for backward compatibility but emits a :class:`DeprecationWarning`
   and internally depends on the ``atomscale`` package.

.. toctree::
   :maxdepth: 2

   guides/index
   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
