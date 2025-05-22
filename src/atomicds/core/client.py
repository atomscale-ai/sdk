from __future__ import annotations

import itertools
import os
import platform
import sys
from collections.abc import Callable  # type: ignore[ruleName]
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from importlib.metadata import version
from typing import Any, Literal
from urllib.parse import urljoin

from requests import Session
from requests.adapters import HTTPAdapter
from rich.progress import Progress
from urllib3.util.retry import Retry

__version__ = version("atomicds")


class BaseClient:
    """Base API client implementation"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
    ):
        """
        Args:
            api_key (str | None): API key.
            endpoint (str): Root API endpoint.
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self._session = None

    @property
    def session(self) -> Session:
        """Session under which HTTP requests are issued"""
        if not self._session:
            self._session = self._create_session(self.api_key)
        return self._session  # type: ignore[return-value]

    def _get(
        self,
        sub_url: str,
        params: dict[str, Any] | None = None,
        deserialize: bool = True,
        base_override: str | None = None,
    ) -> list[dict[Any, Any]] | dict[Any, Any] | bytes | None:
        """Method for issuing a GET request

        Args:
            sub_url (str): API sub-url to use.
            params (dict[str, Any] | None): Params to pass in the GET request. Defaults to None.
            deserialize (bool): Whether to JSON deserialize the response data or return raw bytes. Defaults to True.
            base_overrise (str): Base URL to use instead of the default ADS API root URL.

        Raises:
            ClientError: If the response code returned is not within the range of 200-400.

        Returns:
            (list[dict] | dict | bytes | None): Deserialized JSON data or raw bytes. Returns None if response is a 404.

        """
        base_url = base_override or self.endpoint
        response = self.session.get(
            url=urljoin(base_url, sub_url), verify=True, params=params
        )
        if not response.ok:
            if response.status_code == 404:
                return None

            raise ClientError(
                f"Problem retrieving data from {sub_url} with parameters {params}. HTTP Error {response.status_code}: {response.text}"
            )
        if len(response.content) == 0:
            return None

        return response.json() if deserialize else response.content

    def _post_or_put(
        self,
        method: Literal["POST", "PUT"],
        sub_url: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | bytes | None = None,
        headers: dict[str, str] | None = None,
        deserialize: bool = True,
        base_override: str | None = None,
        return_headers: bool = False,
    ) -> list[dict[Any, Any]] | dict[Any, Any] | bytes | None:
        """Method for issuing a POST or PUT request

        Args:
            method (Literal["POST", "PUT"]): Method to use
            sub_url (str): API sub-url to use.
            params (dict[str, Any] | None): Params to pass in the GET request. Defaults to None.
            body (dict[str, Any] | bytes): Body data to send in the POST request.
            headers (dict[str, str] | None): Optional headers to include in the request.
            deserialize (bool): Whether to JSON deserialize the response data or return raw bytes. Defaults to True.
            base_overrise (str): Base URL to use instead of the default ADS API root URL.
            return_headers (bool): Whether to return the headers from the response instead of the content. Defaults
                to False.

        Raises:
            ClientError: If the response code returned is not within the range of 200-400.

        Returns:
            (list[dict] | dict | bytes | None): Deserialized JSON data or raw bytes. Returns None if response is a 404.
        """
        base_url = base_override or self.endpoint
        method_func = self.session.put if method == "PUT" else self.session.post

        # decide whether to use data= (bytes/streams) or json=
        if body is None:
            data_params: dict[str, Any] = {}
        elif isinstance(body, bytes | bytearray):
            data_params = {"data": body}
        elif hasattr(body, "read"):
            # any file-like / RawIOBase
            data_params = {"data": body}
        else:
            # everything else (dict, list, etc.) goes through JSON
            data_params = {"json": body}

        response = method_func(
            url=urljoin(base_url, sub_url),
            verify=True,
            params=params,
            headers=headers or {},
            **data_params,  # type: ignore # noqa: PGH003
        )
        if not response.ok:
            if response.status_code == 404:
                return None

            raise ClientError(
                f"Problem sending data to {sub_url}. HTTP Error {response.status_code}: {response.text}"
            )

        if return_headers:
            return_data: dict[Any, Any] = response.headers  # type: ignore  # noqa: PGH003
        else:
            return_data = response.json() if deserialize else response.content  # type: ignore  #noqa: PGH003

        return return_data

    def _multi_thread(
        self,
        func: Callable[..., Any],
        kwargs_list: list[dict[str, Any]],
        progress_bar: Progress | None = None,
        progress_description: str | None = None,
        progress_kwargs: dict | None = None,
        transient: bool = False,
    ) -> list[Any]:
        """Handles running a function concurrently with a ThreadPoolExecutor

        Arguments:
            func (Callable): Function to run concurrently
            kwargs_list (list): List of keyword argument inputs for the function
            progress_bar (Progress | None): Progress bar to show. Defaults to None.
            progress_description (str | None): Progress bar description.
            progress_kwargs (dict | None): Additional kwargs to pass to the progress task.
            transient (bool): Whether the progress bar is transient. Defaults to False,

        Returns:
            (list[Any]): List of results from passed function in the order of parameters passed
        """
        return_dict = {}

        total_count = len(kwargs_list)
        kwargs_gen = iter(kwargs_list)

        if progress_bar is not None:
            progress_kwargs = progress_kwargs or {"pad": ""}
            task = progress_bar.add_task(
                progress_description or "", total=total_count, **progress_kwargs
            )

        ind = 0
        num_parallel = min(os.cpu_count() or 8, 8)
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            # Get list of initial futures defined by max number of parallel requests
            futures = set()

            for kwargs in itertools.islice(kwargs_gen, num_parallel):
                future = executor.submit(
                    func,
                    **kwargs,
                )

                future.ind = ind  # type: ignore  # noqa: PGH003
                futures.add(future)
                ind += 1

            while futures:
                # Wait for at least one future to complete and process finished
                finished, futures = wait(futures, return_when=FIRST_COMPLETED)

                for future in finished:
                    data = future.result()

                    if progress_bar is not None:
                        progress_bar.update(task, advance=1, refresh=True)  # type: ignore  # noqa: PGH003

                    return_dict[future.ind] = data  # type: ignore  # noqa: PGH003

                # Populate more futures to replace finished
                for kwargs in itertools.islice(kwargs_gen, len(finished)):
                    new_future = executor.submit(
                        func,
                        **kwargs,
                    )

                    new_future.ind = ind  # type: ignore  # noqa: PGH003
                    futures.add(new_future)
                    ind += 1

        if progress_bar is not None and transient:
            progress_bar.remove_task(task)  # type: ignore # noqa: PGH003

        return [t[1] for t in sorted(return_dict.items())]

    @staticmethod
    def _create_session(api_key: str):
        """Create a requests session

        Args:
            api_key (str): API key to include in the header.

        Returns:
            (Session): Requests Session object

        """
        session = Session()
        session.headers = {"X-API-KEY": api_key}

        # User agent information
        atomicds_info = "atomicds/" + __version__
        python_info = f"Python/{sys.version.split()[0]}"
        platform_info = f"{platform.system()}/{platform.release()}"
        session.headers[
            "user-agent"
        ] = f"{atomicds_info} ({python_info} {platform_info})"

        # TODO: Add retry setting to configuration somewhere
        max_retry_num = 3
        retry = Retry(
            total=max_retry_num,
            read=max_retry_num,
            connect=max_retry_num,
            respect_retry_after_header=True,
            status_forcelist=[429, 504, 502],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session


class ClientError(Exception):
    """Generic error thrown by the Atomic Data Sciences API client"""
