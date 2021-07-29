import asyncio
import datetime
import logging
from collections import deque
from typing import Optional, Union

from tornado.ioloop import IOLoop

import dask
from dask.utils import parse_timedelta

from .core import CommClosedError

logger = logging.getLogger(__name__)


class BatchedSend:
    """Batch messages in batches on a stream

    This takes an IOStream and an interval (in ms) and ensures that we send no
    more than one message every interval milliseconds.  We send lists of
    messages.

    Batching several messages at once helps performance when sending
    a myriad of tiny messages.

    Examples
    --------
    >>> stream = yield connect(address)
    >>> bstream = BatchedSend(interval='10 ms')
    >>> bstream.start(stream)
    >>> bstream.send('Hello,')
    >>> bstream.send('world!')

    On the other side, the recipient will get a message like the following::

        ['Hello,', 'world!']
    """

    # XXX why doesn't BatchedSend follow either the IOStream or Comm API?

    def __init__(self, interval, loop=None, serializers=None):
        # XXX is the loop arg useful?
        self.loop = loop or IOLoop.current()
        self.interval = parse_timedelta(interval, default="ms")
        self.waker = asyncio.Event()
        self.stopped = asyncio.Event()
        self._future = None
        self.please_stop = False
        self.buffer = []
        self.comm = None
        self.message_count = 0
        self.batch_count = 0
        self.byte_count = 0
        self.next_deadline = None
        self.recent_message_log = deque(
            maxlen=dask.config.get("distributed.comm.recent-messages-log-length")
        )
        self.serializers = serializers
        self._consecutive_failures = 0

    def start(self, comm):
        self.comm = comm
        try:
            aio_loop = self.loop.asyncio_loop
        except AttributeError:
            self.loop.add_callback(self._background_send)
        else:
            self._future = asyncio.run_coroutine_threadsafe(
                self._background_send(), aio_loop
            )

    def closed(self):
        return self.comm and self.comm.closed()

    def __repr__(self):
        if self.closed():
            return "<BatchedSend: closed>"
        else:
            return "<BatchedSend: %d in buffer>" % len(self.buffer)

    __str__ = __repr__

    async def _background_send(self):
        while not self.please_stop:
            try:
                await asyncio.wait_for(
                    self.waker.wait(),
                    timeout=(
                        self.next_deadline - self.loop.time()
                        if self.next_deadline is not None
                        else None
                    ),
                )
                self.waker.clear()
            except asyncio.TimeoutError:
                pass
            if not self.buffer:
                # Nothing to send
                self.next_deadline = None
                continue
            if self.next_deadline is not None and self.loop.time() < self.next_deadline:
                # Send interval not expired yet
                continue
            payload, self.buffer = self.buffer, []
            self.batch_count += 1
            self.next_deadline = self.loop.time() + self.interval
            try:
                nbytes = await self.comm.write(
                    payload, serializers=self.serializers, on_error="raise"
                )
                if nbytes < 1e6:
                    self.recent_message_log.append(payload)
                else:
                    self.recent_message_log.append("large-message")
                self.byte_count += nbytes
            except CommClosedError as e:
                logger.info("Batched Comm Closed: %s", e)
                break
            except Exception:
                # We cannot safely retry self.comm.write, as we have no idea
                # what (if anything) was actually written to the underlying stream.
                # Re-writing messages could result in complete garbage (e.g. if a frame
                # header has been written, but not the frame payload), therefore
                # the only safe thing to do here is to abort the stream without
                # any attempt to re-try `write`.
                logger.exception("Error in batched write")
                break
            finally:
                payload = None  # lose ref
        else:
            # nobreak. We've been gracefully closed.
            self.stopped.set()
            return

        # If we've reached here, it means `break` was hit above and
        # there was an exception when using `comm`.
        # We can't close gracefully via `.close()` since we can't send messages.
        # So we just abort.
        # This means that any messages in our buffer our lost.
        # To propagate exceptions, we rely on subsequent `BatchedSend.send`
        # calls to raise CommClosedErrors.
        self.stopped.set()
        self.abort()

    def send(self, *msgs):
        """Schedule a message for sending to the other side

        This completes quickly and synchronously
        """
        if self.comm is not None and self.comm.closed():
            raise CommClosedError()

        self.message_count += len(msgs)
        self.buffer.extend(msgs)
        # Avoid spurious wakeups if possible
        if self.next_deadline is None:
            self.waker.set()

    async def close(self, timeout: Optional[Union[float, datetime.timedelta]] = None):
        """Flush existing messages and then close comm

        If set, raises `asyncio.TimeoutError` after a timeout.
        """
        if self.comm is None:
            return
        self.please_stop = True
        self.waker.set()
        if isinstance(timeout, datetime.timedelta):
            timeout = timeout.total_seconds()
        try:
            await asyncio.wait_for(self.stopped.wait(), timeout=timeout)
        finally:
            if self._future:
                self._future.cancel()
        if not self.comm.closed():
            try:
                if self.buffer:
                    self.buffer, payload = [], self.buffer
                    await self.comm.write(
                        payload, serializers=self.serializers, on_error="raise"
                    )
            except CommClosedError:
                pass
            await self.comm.close()

    def abort(self):
        if self.comm is None:
            return
        self.please_stop = True
        self.buffer = []
        self.waker.set()
        if not self.comm.closed():
            self.comm.abort()
