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
