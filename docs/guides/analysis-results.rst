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

Low-Level Features
------------------

RHEED videos expose a larger set of low-level, per-region features (e.g.
``area_0``, ``eccentricity_0``, ``fwhm_0_3``) beyond the standard columns above.
Request them with :meth:`~atomscale.client.Client.get_rheed_timeseries`, which
returns a DataFrame indexed by ``["Angle", "Frame Number"]``:

.. code-block:: python

   df = client.get_rheed_timeseries(data_id, include_low_level_features=True)
   print(df.filter(like="area").columns)

The low-level columns keep their raw backend names (they are not renamed).

Segmentation Masks
------------------

Each *featurized* frame of a processed RHEED video carries a binary segmentation
mask of the diffraction pattern. Attach the masks to the timeseries — aligned on
the ``Frame Number`` axis, alongside any low-level features — with
``include_masks``:

.. code-block:: python

   from atomscale.results import decode_mask_rle

   df = client.get_rheed_timeseries(
       data_id,
       include_low_level_features=True,
       include_masks=True,
   )

   # Mask columns: mask_rle (COCO RLE string), mask_height, mask_width. Coverage
   # is sparse -- frames without a mask are NA -- so drop those rows first.
   row = df.dropna(subset=["mask_rle"]).iloc[0]
   mask = decode_mask_rle(row["mask_rle"], row["mask_height"], row["mask_width"])
   print(mask.shape)  # (H, W) uint8, values 0/1

Fetch masks on their own — optionally decoded and keyed by absolute frame
number — with :meth:`~atomscale.client.Client.get_frame_masks`:

.. code-block:: python

   masks = client.get_frame_masks(data_id, decode=True)  # {frame_number: (H, W) array}

Embedding Vectors
-----------------

The similarity pipeline persists Chronos embedding vectors for RHEED data — the
*inputs* to similarity matching, as opposed to the derived similarity-vs-time
trajectory. Fetch them with :meth:`~atomscale.client.Client.get_embeddings`:

.. code-block:: python

   emb = client.get_embeddings(data_id, window_span=60.0, kind="window")
   print(emb.vectors.shape)  # (n_windows, dimension)

To find the RHEED data items most similar to a given one, run a
k-nearest-neighbour query over the embedding index with
:meth:`~atomscale.client.Client.query_rheed_embeddings`:

.. code-block:: python

   neighbours = client.query_rheed_embeddings(data_id, top_k=10)
   print(neighbours[["data_id", "similarity"]])

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
