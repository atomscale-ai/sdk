Analysis Results
================

Access the data extracted from your RHEED videos and images.

Fetch Results
-------------

.. code-block:: python

   from atomscale import Client

   client = Client()

   # Search for data
   search_results = client.search(keywords=["GaN"], status="success")

   # Fetch analysis results
   analysed = client.get(search_results["Data ID"].to_list())

Each item in ``analysed`` is a result object with properties for accessing
different types of analysis data.

Timeseries Data
---------------

For RHEED videos, get frame-by-frame analysis:

.. code-block:: python

   video = analysed[0]
   df = video.timeseries_data

   print(df.columns)
   print(df.tail())

Common columns:

.. list-table::
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

Extracted Frames
----------------

Access snapshots extracted during analysis:

.. code-block:: python

   snapshot = video.snapshot_image_data[0]

   # Get matplotlib figure
   fig = snapshot.get_plot()
   fig.savefig("snapshot.png")

   # Get diffraction pattern as DataFrame
   pattern_df = snapshot.get_pattern_dataframe()

   # Get pattern as NetworkX graph
   graph = snapshot.pattern_graph

Result Types
------------

The type of result object depends on the source data:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Data Type
     - Result Class
   * - ``rheed_stationary``, ``rheed_rotating``
     - :class:`~atomscale.results.rheed_video.RHEEDVideoResult`
   * - ``rheed_image``
     - :class:`~atomscale.results.rheed_image.RHEEDImageResult`
   * - ``xps``
     - :class:`~atomscale.results.xps.XPSResult`

Batch Processing
----------------

Process multiple results efficiently:

.. code-block:: python

   for item in analysed:
       if hasattr(item, "timeseries_data"):
           df = item.timeseries_data
           avg_intensity = df["specular_intensity"].mean()
           print(f"{item.data_id}: avg intensity = {avg_intensity:.2f}")
