Monitor Live Results
====================

Poll for analysis results in real-time during active streaming sessions.

When to Use This
----------------

- Watch analysis results arrive during a live RHEED stream
- Update a dashboard or plot as new data comes in
- Trigger actions when certain conditions are met

.. tip::

   For already-completed data, use :doc:`analysis-results` instead.

Choose Your Approach
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Approach
     - Use When
   * - Synchronous loop
     - Simple scripts that can block
   * - Background thread
     - GUI apps or acquisition loops that can't block
   * - Async iterator
     - Asyncio applications

Synchronous Polling
-------------------

Block and wait for each update:

.. code-block:: python

   from atomscale import Client
   from atomscale.timeseries.polling import iter_poll

   client = Client()
   data_id = "YOUR_STREAM_DATA_ID"

   for result in iter_poll(
       client,
       data_id=data_id,
       interval=5.0,      # Poll every 5 seconds
       last_n=10,         # Get last 10 rows
       max_polls=20,      # Stop after 20 polls
   ):
       print(f"Latest timestamp: {result['timestamp'].iloc[-1]}")
       print(result.tail())

Background Thread
-----------------

Poll without blocking your main code:

.. code-block:: python

   from atomscale import Client
   from atomscale.timeseries.polling import start_polling_thread

   client = Client()
   collected = []

   def on_new_data(result):
       print(f"Got {len(result)} rows")
       collected.append(result)

   stop_event = start_polling_thread(
       client,
       data_id="YOUR_STREAM_DATA_ID",
       interval=5.0,
       last_n=10,
       max_polls=20,
       on_result=on_new_data,
   )

   # Your main code continues here...

   # To stop early:
   # stop_event.set()

.. caution::

   The callback runs in the polling thread. Use ``queue.Queue`` or other
   thread-safe mechanisms to update UI elements.

Async Polling
-------------

For asyncio applications:

.. code-block:: python

   import asyncio
   from atomscale import Client
   from atomscale.timeseries.polling import aiter_poll

   client = Client()

   async def watch_stream():
       async for result in aiter_poll(
           client,
           data_id="YOUR_STREAM_DATA_ID",
           interval=5.0,
           last_n=10,
           max_polls=20,
       ):
           print(f"Got {len(result)} rows")

   asyncio.run(watch_stream())

Avoid Duplicate Processing
--------------------------

Use ``distinct_by`` to skip results you've already seen:

.. code-block:: python

   def latest_timestamp(df):
       if df.empty or "timestamp" not in df.columns:
           return None
       return df["timestamp"].iloc[-1]

   for result in iter_poll(
       client,
       data_id=data_id,
       interval=5.0,
       distinct_by=latest_timestamp,  # Only yield when timestamp changes
   ):
       process(result)

Similarity Trajectory Polling
-----------------------------

For similarity trajectory data (structure-property relationships), use the
dedicated trajectory polling functions. These auto-stop when the trajectory
completes:

.. code-block:: python

   from atomscale.similarity.polling import iter_poll_trajectory

   for result in iter_poll_trajectory(
       client,
       source_id="YOUR_DATA_ID",  # or physical_sample_id
       interval=5.0,
       last_n=10,
   ):
       print(f"Active: {result['Active'].any()}")
       if not result["Active"].any():
           print("Trajectory complete")

The trajectory functions mirror the timeseries API:

- ``iter_poll_trajectory()`` - Synchronous loop
- ``start_polling_trajectory_thread()`` - Background thread
- ``aiter_poll_trajectory()`` - Async iterator
- ``start_polling_trajectory_task()`` - Async background task
