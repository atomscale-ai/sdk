Poll Similarity Trajectory
==========================

This guide covers the polling helpers in :mod:`atomscale.similarity.polling` for
monitoring similarity trajectory data. These functions mirror the timeseries
polling API but are tailored for trajectory workflows:

- **Manual loop** with :func:`iter_poll_trajectory` – ideal for scripts that can
  block until each poll finishes.
- **Background thread** with :func:`start_polling_trajectory_thread` – keeps
  polling while your main thread keeps working.
- **Async iterator** with :func:`aiter_poll_trajectory` – awaits each update
  inside an async function.
- **Async background task** with :func:`start_polling_trajectory_task` –
  fire-and-forget inside an asyncio application.

.. tip::

   If you're polling generic timeseries data (e.g., RHEED intensity), see
   :doc:`poll-timeseries` instead. Use this guide for similarity trajectory
   workflows that track structure-property relationships across experiments.

Key differences from timeseries polling
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Timeseries polling
     - Trajectory polling
   * - Identifier
     - ``data_id``
     - ``source_id`` (data_id or physical_sample_id)
   * - Stop condition
     - Manual (``max_polls`` or custom ``until``)
     - Auto-stops when ``Active`` is ``False``
   * - Return format
     - DataFrame with timeseries data
     - DataFrame with ``Active`` column

.. important::

   **Default stop condition**: Polling stops automatically when no trajectory is
   active (``not df["Active"].any()``). Override this behavior with the ``until``
   parameter if you need custom termination logic.

Shared setup
------------

.. code-block:: python

   from atomscale.client import Client
   from atomscale.similarity.polling import (
       iter_poll_trajectory,
       aiter_poll_trajectory,
       start_polling_trajectory_thread,
       start_polling_trajectory_task,
   )

   client = Client(api_key="YOUR_API_KEY")
   source_id = "YOUR_SOURCE_ID"  # data_id or physical_sample_id

Synchronous polling
-------------------

Loop over :func:`iter_poll_trajectory` to fetch trajectory updates on a fixed
cadence. Polling stops automatically when the trajectory is no longer active,
or you can set ``max_polls`` to limit iterations.

.. code-block:: python

   for idx, result in enumerate(
       iter_poll_trajectory(
           client,
           source_id=source_id,
           interval=5.0,
           last_n=10,
           max_polls=10,
       ),
       start=1,
   ):
       print(f"Poll {idx}: {len(result)} rows, active={result['Active'].any()}")
       print(result.tail())

.. note::

   The iterator uses **drift-corrected scheduling** to maintain accurate timing
   even when individual polls are slow.

Background thread helper
------------------------

Use :func:`start_polling_trajectory_thread` when you want updates but cannot
block the main thread (e.g., inside a GUI or acquisition loop). The helper
spawns a daemon thread and forwards each update to your callback.

.. code-block:: python

   collected = []


   def on_result(result):
       print(f"Thread received {len(result)} rows")
       collected.append(result)


   stop_event = start_polling_trajectory_thread(
       client,
       source_id=source_id,
       interval=10.0,
       last_n=10,
       on_result=on_result,
   )

   # Polling stops automatically when trajectory is inactive.
   # Call stop_event.set() to terminate early.

.. caution::

   The callback runs in the polling thread, not the main thread. If you need to
   update UI elements, use thread-safe mechanisms (e.g., ``queue.Queue``).

Async utilities
---------------

Two helpers integrate with asyncio:

* :func:`aiter_poll_trajectory` yields results without blocking the event loop.
* :func:`start_polling_trajectory_task` creates a background task that invokes
  an (optional) async handler for each result.

**Async iterator**

.. code-block:: python

   import asyncio


   async def stream_updates():
       async for result in aiter_poll_trajectory(
           client,
           source_id=source_id,
           interval=5.0,
           last_n=10,
       ):
           print(f"Async poll received {len(result)} rows")
           if not result["Active"].any():
               print("Trajectory complete")


   asyncio.run(stream_updates())

**Background task**

.. code-block:: python

   async def handle_async(result):
       print(f"Task handler received {len(result)} rows")


   async def main():
       task = start_polling_trajectory_task(
           client,
           source_id=source_id,
           interval=5.0,
           last_n=5,
           on_result=handle_async,
       )
       await task


   asyncio.run(main())

.. tip::

   Cancel the task with ``task.cancel()`` to stop polling early in async code.

Custom stop conditions
----------------------

Override the default stop condition with the ``until`` parameter:

.. code-block:: python

   # Stop after receiving at least 100 data points
   for result in iter_poll_trajectory(
       client,
       source_id=source_id,
       interval=5.0,
       until=lambda df: len(df) >= 100,
   ):
       print(f"Received {len(result)} rows")

Error handling
--------------

Use the ``on_error`` callback to handle transient failures without stopping the
poll loop:

.. code-block:: python

   def handle_error(exc):
       print(f"Poll failed: {exc}, retrying...")


   for result in iter_poll_trajectory(
       client,
       source_id=source_id,
       interval=5.0,
       on_error=handle_error,
   ):
       print(f"Received {len(result)} rows")

.. warning::

   Errors are swallowed when ``on_error`` is provided. If you need to fail fast,
   omit the callback and handle exceptions in your loop.
