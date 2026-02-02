Stream Instrument Data
======================

Stream scalar timeseries data (temperature, pressure, power, etc.) from growth
instruments to Atomscale in real-time.

When to Use This
----------------

- Log instrument parameters during growth
- Correlate sensor data with RHEED analysis
- Monitor multiple channels simultaneously

Create a Streamer
-----------------

.. code-block:: python

   from atomscale.streaming import TimeseriesStreamer

   streamer = TimeseriesStreamer(api_key="YOUR_API_KEY")

Initialize a Stream
-------------------

.. code-block:: python

   data_id = streamer.initialize(stream_name="Growth Run 001")

Link to a Growth Instrument (Optional)
--------------------------------------

If you've configured growth instruments in Atomscale, you can link streams to
them:

.. code-block:: python

   from atomscale import Client

   client = Client()

   # List your instruments
   instruments = client.list_growth_instruments()
   for inst in instruments:
       print(f"{inst['synth_source_id']}: {inst['source_name']}")

   # Initialize with instrument link
   data_id = streamer.initialize(
       stream_name="Growth Run 001",
       synth_source_id=instruments[0]["synth_source_id"],
   )

Push Single Channel
-------------------

Send data for one channel at a time:

.. code-block:: python

   import time

   for chunk_idx in range(10):
       t = time.time()
       timestamps = [t + i * 0.1 for i in range(100)]
       values = [580.0 + i * 0.1 for i in range(100)]

       streamer.push(
           data_id=data_id,
           chunk_index=chunk_idx,
           channel_name="substrate_temp",
           timestamps=timestamps,
           values=values,
           units="C",
       )
       time.sleep(1.0)

   streamer.finalize(data_id)

Push Multiple Channels
----------------------

Send multiple channels in a single call:

.. code-block:: python

   import time

   for chunk_idx in range(10):
       t = time.time()
       timestamps = [t + i * 0.1 for i in range(100)]

       streamer.push_multi(
           data_id=data_id,
           chunk_index=chunk_idx,
           channels={
               "substrate_temp": {
                   "timestamps": timestamps,
                   "values": [580.0 + i * 0.1 for i in range(100)],
                   "units": "C",
               },
               "chamber_pressure": {
                   "timestamps": timestamps,
                   "values": [2e-9 + i * 1e-12 for i in range(100)],
                   "units": "Torr",
               },
           },
       )
       time.sleep(1.0)

   streamer.finalize(data_id)

Run Mode (Iterator)
-------------------

Stream from an iterator when data is already buffered:

.. code-block:: python

   def data_generator():
       for chunk_idx in range(10):
           timestamps = [chunk_idx * 100 + i for i in range(100)]
           values = [25.0 + i * 0.1 for i in range(100)]
           yield (timestamps, values)

   streamer.run(
       data_id=data_id,
       channel_name="temperature",
       data_iter=data_generator(),
       units="C",
   )
   streamer.finalize(data_id)

.. note::

   ``run()`` blocks until all chunks upload. Use ``push()`` for non-blocking
   uploads.

Best Practices
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Practice
     - Why
   * - Always call ``finalize()``
     - Signals stream is complete so processing can begin
   * - Use consistent ``chunk_index``
     - Ensures data is ordered correctly
   * - Include ``units``
     - Makes data easier to interpret in the UI

.. warning::

   Failing to call ``finalize()`` leaves the stream incomplete.
