Stream RHEED Video
==================

Stream live RHEED frames directly from your instrument to Atomscale. Analysis
runs in real-time as data arrives.

When to Use This
----------------

- Your camera SDK provides frames as numpy arrays
- You want real-time analysis during growth
- You need results before the growth completes

.. tip::

   For pre-recorded videos, use :doc:`upload-files` instead.

Create a Streamer
-----------------

.. code-block:: python

   from atomscale.streaming import RHEEDStreamer

   streamer = RHEEDStreamer(api_key="YOUR_API_KEY")

Initialize a Stream
-------------------

.. code-block:: python

   data_id = streamer.initialize(
       fps=120.0,
       rotations_per_min=15.0,  # Use 0.0 for stationary RHEED
       chunk_size=240,          # Frames per chunk (2 seconds at 120fps)
       stream_name="Growth 001",
       physical_sample="GaN-2025-001",
   )

The ``rotations_per_min`` parameter determines whether data is classified as
rotating or stationary RHEED.

Link to a Physical Sample
-------------------------

The ``physical_sample`` parameter accepts either a **name** or a **UUID**:

**By name** (recommended for new samples):

.. code-block:: python

   data_id = streamer.initialize(
       fps=120.0,
       rotations_per_min=15.0,
       chunk_size=240,
       physical_sample="Wafer-A1",  # Matched case-insensitively; created if not found
   )

**By UUID** (for existing samples):

.. code-block:: python

   data_id = streamer.initialize(
       fps=120.0,
       rotations_per_min=15.0,
       chunk_size=240,
       physical_sample="a1b2c3d4-e5f6-7890-abcd-ef1234567890",  # Must exist
   )

When you also provide a ``project_id``, the project's ``tracking_physical_sample_id``
configuration is automatically updated to link the sample for growth monitoring:

.. code-block:: python

   data_id = streamer.initialize(
       fps=120.0,
       rotations_per_min=15.0,
       chunk_size=240,
       physical_sample="Wafer-A1",
       project_id="your-project-uuid",
   )

Push Mode (Live Capture)
------------------------

Use when your camera SDK delivers frames in real-time:

.. code-block:: python

   import numpy as np
   import time

   fps = 120.0
   chunk_size = 240
   seconds_per_chunk = chunk_size / fps

   for chunk_idx in range(num_chunks):
       # Get frames from your camera (shape: [chunk_size, height, width])
       frames = camera.capture(chunk_size)  # Your camera SDK

       streamer.push(data_id, chunk_idx, frames)
       time.sleep(seconds_per_chunk)

   streamer.finalize(data_id)

Run Mode (Buffered Data)
------------------------

Use when frames are already in memory:

.. code-block:: python

   import numpy as np
   import time

   def frame_generator(all_frames, chunk_size, fps):
       """Yield chunks at the correct pace."""
       for start in range(0, len(all_frames), chunk_size):
           yield all_frames[start:start + chunk_size]
           time.sleep(chunk_size / fps)

   # Frames already in memory
   frames = np.load("recorded_frames.npy")

   data_id = streamer.initialize(
       fps=120.0,
       rotations_per_min=0.0,
       chunk_size=240,
       stream_name="Playback",
   )

   streamer.run(data_id, frame_generator(frames, chunk_size=240, fps=120.0))
   streamer.finalize(data_id)

Uploads and the Disk Spool
--------------------------

``push()`` and ``run()`` never wait on the network. Each chunk is stamped,
encoded and written to a spool directory on disk, and background workers
upload the spool oldest first. A slow or failing connection never stalls your
acquisition loop, so it never delays your chunks' timestamps either. A failed
upload is retried with backoff (up to 30 s apart, with a fresh upload URL each
time) until it succeeds, the stream closes, or the chunk is older than the
retention limit.

.. code-block:: python

   streamer = RHEEDStreamer(
       api_key="YOUR_API_KEY",
       spool_dir="D:/rheed-spool",          # default: your user cache directory
       spool_max_bytes=50_000_000_000,      # default: min(20 GB, 10% of free disk)
       spool_min_free_bytes=2_000_000_000,  # default: 1 GB
   )

   ...
   report = streamer.finalize(data_id)
   print(report["chunks_uploaded"], report["chunks_not_uploaded"])

- **Bounds.** When the spool reaches ``spool_max_bytes``, or the disk would fall
  below ``spool_min_free_bytes`` free, the incoming chunk is **dropped** rather
  than blocking. It is logged and counted in ``spool_status()`` and in the
  ``finalize()`` report, and the platform sees a missing chunk.
- **Finalize.** ``finalize()`` first waits (up to ``drain_timeout_s``, default
  10 minutes) for the stream's spool to upload. It then returns a report of
  what became of every chunk and warns if any was never uploaded.
- **Crashes.** Spooled chunks survive a crash. The next ``RHEEDStreamer`` for
  the same endpoint uploads them in the background if their stream is still
  open, and deletes them if the stream was finalized, has had no upload for
  2 hours (the platform has closed it by then), or they are older than
  ``spool_retention_s`` (default 24 hours).
- **Location.** One subfolder per stream under ``spool_dir``, or the
  ``ATOMSCALE_SPOOL_DIR`` environment variable, or
  ``<user cache dir>/atomscale/rheed-spool``.

Frame Requirements
------------------

- **Data type**: ``uint8`` grayscale
- **Shape**: ``(N, height, width)`` for chunks, ``(height, width)`` for single frames
- **Timing**: Maintain real-time pacing for accurate analysis

Best Practices
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Practice
     - Why
   * - Chunk size >= 2 seconds
     - Balances upload overhead with latency
   * - Maintain capture cadence
     - Server expects real-time pacing
   * - Always call ``finalize()``
     - Signals stream is complete so processing can finish
   * - Use descriptive ``stream_name``
     - Makes data easier to find later

.. warning::

   Failing to call ``finalize()`` leaves the stream incomplete and may prevent
   analysis from completing.
