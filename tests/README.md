# Running tests

Install the package with its development dependencies (`uv pip install -e '.[dev]'`),
then run `pytest`. No API credentials or remote service is needed. Localhost sockets
must be available for the HTTP and Rust streaming tests.

`conftest.py` creates a fresh `Sandbox` and client for each test. `sandbox.py` supplies
synthetic catalogue entries and complete responses for spectra, RHEED timeseries,
optical snapshots, tool state, recipes, ellipsometry, and similarity/changepoints.
The actual client and provider code constructs results; only `_get` is replaced.
Responses are deep copies, unknown routes fail, and no test must run before another.
Search tests check the outgoing query contract rather than pretending to implement
server filtering. A localhost HTTP test also covers client serialization and hydration.

Offline tests clear API credentials and block external Python requests, including
redirects. Loopback remains available. Rust streaming tests use their own explicit
localhost servers; the Python request guard does not intercept Rust networking.
Polling threads are stopped and joined before fixture teardown.

When adding a result type, add its catalogue entry and endpoint responses to
`Sandbox`, then add it to `test_client.py`'s parametrized hydration test. Keep the
fixture populated enough to exercise assertions; missing synthetic data should fail,
not skip. Empty and malformed payloads belong in separate focused tests.

## Live checks

Live tests are marked `live` and skipped by default, even if API credentials exist.
Run `pytest --run-live -m live` against a **dedicated test environment** with:

- `AS_TEST_API_ENDPOINT` and `AS_TEST_API_KEY` (no production-default fallback).
- `AS_TEST_RHEED_IMAGE_ID`: a processed image with a populated fingerprint suitable
  for plotting, Laue geometry, and feature extraction.
- `AS_TEST_PHYSICAL_SAMPLE_ID`: a sample with finite `rheed_quality` timeseries.
- `AS_TEST_SIMILARITY_SOURCE_ID`: an embedding source with at least one neighbor.

These are pinned records, not the first items returned from an account's catalogue.
An explicitly enabled live check fails if configuration or expected data is missing.
The tests only read these records; they do not provision a remote environment or
upload a dataset.

The normal CI matrix runs offline. The separate **Live API checks** workflow is
manually dispatched and uses the GitHub environment `sdk-test`: configure the key
as an environment secret and the endpoint/IDs as environment variables there.

## Branch fixes reviewed

- `origin/fix/ci-rust-cache-and-http-timeout`: the polling ID indexing and sample
  name-mask fixes identified real issues. Local fixtures remove the polling setup
  dependency; pinned samples remove the fragile catalogue scan entirely.
- `origin/mikep-atomscale/disable-personal-search-tests`: personal catalogue skips
  avoid empty-account failures, but deterministic query tests retain coverage.
- `origin/bugfix/missing_data_in_tests`, `origin/bugfix/testing`, and
  `origin/ci/auth-preflight-hardening`: success filtering, bounded requests, and auth
  diagnostics informed the review. Ordinary tests now have no live catalogue/auth
  dependency. The older `scripts/ci_auth_preflight.py` remains a standalone diagnostic
  for the legacy live corpus; it is no longer a prerequisite for CI.
