Poll Time Series Updates
========================

``timeseries_polling.ipynb`` shows how to stay current with streaming RHEED
analysis. This guide summarises the four helper entry points in
:mod:`atomscale.timeseries.polling` and explains which "mode" to pick:

- **Manual loop** with :func:`iter_poll` – ideal for scripts that can block
  until each poll finishes.
- **Background thread** with :func:`start_polling_thread` – keeps polling while
  your main thread keeps working.
- **Async iterator** with :func:`aiter_poll` – awaits each update inside an
  async function.
- **Async background task** with :func:`start_polling_task` – fire-and-forget
  inside an asyncio application.

Shared setup
------------

.. code-block:: python

   from atomscale.client import Client
   from atomscale.timeseries.polling import (
       iter_poll,
       aiter_poll,
       start_polling_thread,
       start_polling_task,
   )

   client = Client(api_key="YOUR_API_KEY")
   data_id = "YOUR_TIME_SERIES_DATA_ID"


   def latest_timestamp(df):
       if df.empty or "timestamp" not in df.columns:
           return None
       return df.iloc[-1]["timestamp"]

Synchronous polling
-------------------

Loop over :func:`iter_poll` to fetch fresh rows on a fixed cadence. The helper
waits ``interval`` seconds between polls so a simple ``for`` loop is enough to
keep the script going. Use ``distinct_by`` to avoid duplicates,
``max_polls`` to stop automatically, and ``fire_immediately=False`` to skip the
first immediate request if you only want timed polls.

.. code-block:: python

   for idx, result in enumerate(
       iter_poll(
           client,
           data_id=data_id,
           interval=5.0,
           last_n=10,
           distinct_by=latest_timestamp,
           max_polls=3,
       ),
       start=1,
   ):
       print(f"Poll {idx}: latest timestamp -> {latest_timestamp(result)}")
       print(result.tail())

Background thread helper
------------------------

Use :func:`start_polling_thread` when you want updates but cannot block the
main thread (for example, inside a GUI or acquisition loop). The helper spawns
a daemon thread, starts polling immediately, and forwards each update to your
callback. Call ``stop_event.set()`` to shut it down cleanly.

.. code-block:: python

   collected = []


   def on_result(result):
       print(f"Thread received {len(result)} rows")
       collected.append(result)


   stop_event = start_polling_thread(
       client,
       data_id=data_id,
       interval=10.0,
       last_n=10,
       max_polls=5,
       distinct_by=latest_timestamp,
       on_result=on_result,
   )

   # Call stop_event.set() to terminate early.

Async utilities
---------------

Two helpers integrate with asyncio:

* :func:`aiter_poll` yields results without blocking the event loop, so you can
  ``async for`` over updates.
* :func:`start_polling_task` creates a background task that awaits the poller
  in parallel and invokes an (optional) async handler for each result.

.. code-block:: python

   import asyncio


   async def stream_updates():
       async for result in aiter_poll(
           client,
           data_id=data_id,
           interval=5.0,
           last_n=10,
           distinct_by=latest_timestamp,
           max_polls=3,
       ):
           print(f"Async poll received {len(result)} rows")
           print(result.tail())


   asyncio.run(stream_updates())

.. code-block:: python

   async def handle_async(result):
       print(f"Task handler received {len(result)} rows")


   async def main():
       task = start_polling_task(
           client,
           data_id=data_id,
           interval=5.0,
           last_n=5,
           max_polls=3,
           distinct_by=latest_timestamp,
           on_result=handle_async,
       )
       await task


   asyncio.run(main())
