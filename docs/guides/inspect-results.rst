Inspect Analysis Results
========================

``general_use.ipynb`` demonstrates how to work with objects returned by
:meth:`atomscale.client.Client.get`. This guide walks through the same workflow
in a linear format.

Fetch analysed items
--------------------

.. code-block:: python

   from atomscale.client import Client

   client = Client(api_key="YOUR_API_KEY")
   search_results = client.search(keywords="demo", data_type="rheed_stationary")

   analysed = client.get(search_results["Data ID"].to_list())

Each item in ``analysed`` is a subclass of
:class:`atomscale.results.RHEEDVideoResult` or
:class:`atomscale.results.RHEEDImageResult`, depending on the source data.

.. note::

   The :meth:`get` call fetches metadata and analysis artefacts for each ID.
   For large result sets, consider batching or filtering first.

Inspect time series data
------------------------

.. code-block:: python

   video_item = analysed[0]
   timeseries = video_item.timeseries_data
   print(timeseries.columns)
   print(timeseries.tail())

The timeseries frame contains specular intensity, strain metrics, cluster IDs,
and other summary features for every frame in the video.

.. list-table:: Common timeseries columns
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``timestamp``
     - Frame timestamp in seconds
   * - ``specular_intensity``
     - Specular spot brightness
   * - ``strain``
     - Computed strain metric
   * - ``cluster_id``
     - Pattern cluster assignment

Work with extracted frames
--------------------------

.. code-block:: python

   snapshot = video_item.snapshot_image_data[0]
   figure = snapshot.get_plot()  # Matplotlib figure
   fingerprint = snapshot.pattern_graph
   df = snapshot.get_pattern_dataframe()

``pattern_graph`` exposes the detected diffraction network as a NetworkX graph,
while :meth:`get_pattern_dataframe` returns a tidy table describing each spot.

.. tip::

   Use ``figure.savefig("snapshot.png")`` to export plots for reports or
   publications.

Download processed videos
-------------------------

.. code-block:: python

   client.download_videos(
       data_ids=search_results["Data ID"].to_list(),
       dest_dir="processed/",
   )

The files are saved as MP4 (one per data ID) and mirror what you see in the UI.

.. caution::

   Downloaded videos can be large. Ensure you have sufficient disk space and
   consider filtering to only the IDs you need.

.. seealso::

   - :doc:`poll-timeseries` – Poll for live timeseries updates
   - :doc:`poll-trajectory` – Poll similarity trajectory data
