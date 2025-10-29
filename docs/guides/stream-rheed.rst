Stream RHEED Video
==================

``rheed_streaming.ipynb`` demonstrates both push (callback) and pull (generator)
streaming with :class:`atomicds.streaming.rheed_stream.RHEEDStreamer`. This
guide condenses the notebook into a quick reference and explains when to choose
each style:

- **Callback / push mode** – the camera or SDK hands you fresh frames and you
  upload each chunk immediately.
- **Generator / pull mode** – you already have frames buffered (from disk,
  memory, or a simulated source) and want the helper to pace the upload for you.

Prerequisites
-------------

- ``numpy`` installed
- RHEED frames as ``uint8`` arrays shaped ``(N, H, W)`` or ``(H, W)``
- A stable clock so you can honour the capture cadence

Create a streamer
-----------------

.. code-block:: python

   from atomicds.streaming.rheed_stream import RHEEDStreamer

   streamer = RHEEDStreamer(api_key="YOUR_API_KEY")

Optional keyword arguments tune chunking and logging. For example,
``verbosity=4`` emits detailed progress, and ``max_workers`` caps concurrency.
If you already know the sample name, pass it to :meth:`initialize` so the data
links to the right physical sample (names are matched case-insensitively or
created on the fly).

Callback / push mode
--------------------

Use this variant when frames arrive live from the instrument. The outer loop is
your acquisition callback: once a chunk is ready, send it to the API and wait
just long enough to match the capture cadence.

.. code-block:: python

   import numpy as np
   import time

   fps = 120.0
   chunk_size = 240  # ≥ 2 seconds of frames is recommended
   seconds_per_chunk = chunk_size / fps

   data_id = streamer.initialize(
       fps=fps,
       rotations_per_min=15.0,  # set to 0.0 for stationary
       chunk_size=chunk_size,
       stream_name="Demo (callback mode)",
       physical_sample="Demo wafer",
   )

   for chunk_idx in range(5):
       frames = np.random.randint(0, 256, size=(chunk_size, 300, 500), dtype=np.uint8)
       streamer.push(data_id, chunk_idx, frames)
       time.sleep(seconds_per_chunk)

   time.sleep(1.0)  # let in-flight uploads finish
   streamer.finalize(data_id)

Generator / pull mode
---------------------

Use this form when frames are already buffered (for example, saved by the
instrument or simulated offline). Provide an iterator that yields chunks and
the helper will take care of pacing and retry logic.

.. code-block:: python

   def frame_chunks(frames, *, chunk_size=240, fps=120.0):
       seconds_per_chunk = chunk_size / fps
       for start in range(0, len(frames), chunk_size):
           yield frames[start : start + chunk_size]
           time.sleep(seconds_per_chunk)


   frames = np.random.randint(0, 256, size=(1200, 300, 500), dtype=np.uint8)

   data_id = streamer.initialize(
       fps=10.0,
       rotations_per_min=0.0,
       chunk_size=20,
       stream_name="Demo (generator mode)",
       physical_sample="Demo wafer",
   )

   streamer.run(data_id, frame_chunks(frames, chunk_size=20, fps=10.0))
   streamer.finalize(data_id)

Tips
----

- Maintain the original capture cadence so the server can keep up.
- Make each chunk cover at least two seconds of frames.
- Call :meth:`finalize` even if the upload fails part-way; it lets the pipeline
  clean up gracefully.
- Use distinct ``stream_name`` values while testing so you can find runs later.
