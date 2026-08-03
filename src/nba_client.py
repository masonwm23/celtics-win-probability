"""
A thin, polite wrapper around nba_api endpoint calls.

Why this exists: stats.nba.com is intermittent. It will occasionally time out,
return an empty payload, or drop a connection for no reason. Without a retry
layer, a long pull produces silent gaps that are very hard to notice later.

The contract here is deliberate:
  - Retry a fixed number of times with increasing backoff.
  - Sleep briefly between successful calls so we do not hammer the endpoint.
  - If every attempt fails, RAISE. Callers are responsible for recording the
    failure in a manifest. Nothing in this project may quietly return empty
    data on failure, because that is indistinguishable from a game with no
    events.
"""

import time
import logging

from src import config

logger = logging.getLogger(__name__)


class NBARequestError(RuntimeError):
    """Raised when an endpoint call fails after all retries."""


def call_endpoint(endpoint_cls, **kwargs):
    """
    Instantiate an nba_api endpoint class with retries and return the object.

    Parameters
    ----------
    endpoint_cls : class
        An nba_api endpoint class, for example
        ``nba_api.stats.endpoints.playbyplayv3.PlayByPlayV3``.
    **kwargs
        Passed straight to the endpoint. ``timeout`` is injected from config
        unless the caller supplies its own.

    Returns
    -------
    The instantiated endpoint object.

    Raises
    ------
    NBARequestError
        If all attempts fail. The original exception is chained.
    """
    kwargs.setdefault("timeout", config.REQUEST_TIMEOUT)

    last_exc = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            result = endpoint_cls(**kwargs)
            # Touch the payload so a malformed response fails here, inside the
            # retry loop, rather than later in the caller.
            _ = result.get_dict()
            time.sleep(config.REQUEST_DELAY)
            return result
        except Exception as exc:  # noqa: BLE001 - intentionally broad, retried
            last_exc = exc
            wait = config.RETRY_BACKOFF * attempt
            logger.warning(
                "%s attempt %d/%d failed (%s). Retrying in %.1fs",
                endpoint_cls.__name__, attempt, config.MAX_RETRIES,
                type(exc).__name__, wait,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(wait)

    raise NBARequestError(
        f"{endpoint_cls.__name__} failed after {config.MAX_RETRIES} attempts "
        f"with kwargs {kwargs}: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc
