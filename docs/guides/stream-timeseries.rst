Stream Instrument Timeseries
============================

The :class:`~atomscale.streaming.rheed_stream.TimeseriesStreamer` lets you
stream scalar timeseries data (temperature, pressure, power, etc.) from
instruments to Atomscale in real-time.

Like RHEED streaming, there are two modes:

- **Push mode** – Send data as it arrives from your instruments.
- **Run mode** – Stream from an iterator when data is already buffered.

Create a streamer
-----------------

.. code-block:: python

   from atomscale.streaming import TimeseriesStreamer

   streamer = TimeseriesStreamer(api_key="YOUR_API_KEY")

Initialize a stream
-------------------

Before sending data, initialize a stream to get a ``data_id``:

.. code-block:: python

   data_id = streamer.initialize(
       stream_name="Growth Run 1",
       instrument_type="mbe",
   )

Push mode (single channel)
--------------------------

Use ``push()`` when data arrives live from instruments. Each call sends one
chunk for one channel:

.. code-block:: python

   import time

   # Stream temperature data
   for chunk_idx in range(10):
       timestamps = [time.time() + i * 0.1 for i in range(100)]
       values = [25.0 + i * 0.01 for i in range(100)]

       streamer.push(
           data_id=data_id,
           chunk_index=chunk_idx,
           channel_name="temperature",
           timestamps=timestamps,
           values=values,
           units="C",
       )
       time.sleep(1.0)

   streamer.finalize(data_id)

Push mode (multiple channels)
-----------------------------

Use ``push_multi()`` to send multiple channels in one call:

.. code-block:: python

   import time

   for chunk_idx in range(10):
       t = time.time()
       timestamps = [t + i * 0.1 for i in range(100)]

       streamer.push_multi(
           data_id=data_id,
           chunk_index=chunk_idx,
           channels={
               "temperature": {
                   "timestamps": timestamps,
                   "values": [25.0 + i * 0.01 for i in range(100)],
                   "units": "C",
               },
               "pressure": {
                   "timestamps": timestamps,
                   "values": [1e-6 + i * 1e-9 for i in range(100)],
                   "units": "mbar",
               },
           },
       )
       time.sleep(1.0)

   streamer.finalize(data_id)

Run mode (iterator)
-------------------

Use ``run()`` when you have buffered data. Provide an iterator that yields
``(timestamps, values)`` tuples:

.. code-block:: python

   def data_generator():
       """Yield chunks of (timestamps, values)."""
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

   ``run()`` blocks until all chunks are uploaded. Use ``push()`` for
   non-blocking uploads.

Complete example
----------------

.. code-block:: python

   import time
   from atomscale.streaming import TimeseriesStreamer

   # Create streamer
   streamer = TimeseriesStreamer(api_key="YOUR_API_KEY")

   # Initialize stream
   data_id = streamer.initialize(
       stream_name="MBE Growth - Sample A",
       instrument_type="mbe",
   )

   # Stream data for 10 seconds
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
               "ga_flux": {
                   "timestamps": timestamps,
                   "values": [1.2e-7 + i * 1e-10 for i in range(100)],
                   "units": "Torr",
               },
           },
       )
       time.sleep(1.0)

   # Mark stream as complete
   streamer.finalize(data_id)
   print(f"Stream complete: {data_id}")

Best practices
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Practice
     - Reason
   * - Always call ``finalize()``
     - Marks the stream complete so processing can begin
   * - Use consistent ``chunk_index`` values
     - Ensures data is ordered correctly on the server
   * - Include ``units`` when possible
     - Makes data easier to interpret in the UI

.. warning::

   Failing to call ``finalize()`` leaves the stream in an incomplete state.

.. seealso::

   - :doc:`stream-rheed` – Stream RHEED video frames
   - :doc:`poll-timeseries` – Monitor analysis results as they arrive
