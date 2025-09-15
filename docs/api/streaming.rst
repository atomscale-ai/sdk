Streaming API
=============

.. automodule:: atomicds.streaming.rheed_stream
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

  from atomicds.streaming.rheed_stream import RHEEDStreamer

  streamer = RHEEDStreamer(api_key="...")
  data_id = streamer.initialize(
      fps=120.0,
      rotations_per_min=0.0,  # stationary
      chunk_size=240,
      stream_name="My RHEED Stream",
  )


  # Generator or iterator yielding (N,H,W) or (H,W) uint8 frames
  def chunk_source():
      # yield your numpy arrays here
      yield ...


  streamer.run(data_id, chunk_source())

  # OR push chunks manually
  for idx, frame_chunk in enumerate(frame_chunks):
      streamer.push(data_id, idx, frame_chunk)

  streamer.finalize(data_id)
