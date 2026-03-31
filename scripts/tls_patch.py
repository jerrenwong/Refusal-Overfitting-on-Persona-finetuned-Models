"""
Replace httpcore's anyio backend with a pure-asyncio backend.

anyio's MemoryBIO TLS causes ConnectionResetError with TLS-fingerprinting servers
(e.g. Cloudflare). asyncio.open_connection(ssl=...) works fine because it uses
the OS TLS stack directly.

Import this before any tinker/httpx code.
"""
import asyncio
import socket
import ssl
import typing

import httpcore._backends.anyio as _anyio_mod
from httpcore._backends.base import AsyncNetworkBackend, AsyncNetworkStream
from httpcore._exceptions import ConnectError, ConnectTimeout


class _AsyncioStream(AsyncNetworkStream):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        try:
            coro = self._reader.read(max_bytes)
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except asyncio.TimeoutError:
            raise ConnectTimeout()
        except Exception as exc:
            raise ConnectError() from exc

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return
        self._writer.write(buffer)
        try:
            coro = self._writer.drain()
            if timeout is not None:
                await asyncio.wait_for(coro, timeout=timeout)
            else:
                await coro
        except asyncio.TimeoutError:
            raise ConnectTimeout()
        except Exception as exc:
            raise ConnectError() from exc

    async def aclose(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        # The underlying connection should already be plain TCP.
        # asyncio doesn't support upgrading a StreamWriter to TLS directly,
        # so we use loop.start_tls() on the underlying transport.
        try:
            transport = self._writer.transport
            loop = asyncio.get_running_loop()
            coro = loop.start_tls(
                transport,
                self._writer._protocol,  # type: ignore[attr-defined]
                ssl_context,
                server_side=False,
                server_hostname=server_hostname,
            )
            if timeout is not None:
                new_transport = await asyncio.wait_for(coro, timeout=timeout)
            else:
                new_transport = await coro

            # Patch the writer to use the new TLS transport
            self._writer._transport = new_transport  # type: ignore[attr-defined]
            return self
        except asyncio.TimeoutError:
            raise ConnectTimeout()
        except Exception as exc:
            raise ConnectError() from exc

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object":
            return self._writer.get_extra_info("ssl_object")
        if info == "client_addr":
            return self._writer.get_extra_info("sockname")
        if info == "server_addr":
            return self._writer.get_extra_info("peername")
        if info == "socket":
            return self._writer.get_extra_info("socket")
        return None


class _AsyncioBackend(AsyncNetworkBackend):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable | None = None,
    ) -> AsyncNetworkStream:
        try:
            coro = asyncio.open_connection(host, port)
            if timeout is not None:
                reader, writer = await asyncio.wait_for(coro, timeout=timeout)
            else:
                reader, writer = await coro
            # Apply socket options if any
            if socket_options:
                raw_sock = writer.get_extra_info("socket")
                if raw_sock:
                    for opt in socket_options:
                        raw_sock.setsockopt(*opt)
            return _AsyncioStream(reader, writer)
        except asyncio.TimeoutError:
            raise ConnectTimeout()
        except Exception as exc:
            raise ConnectError() from exc

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable | None = None,
    ) -> AsyncNetworkStream:
        try:
            coro = asyncio.open_unix_connection(path)
            if timeout is not None:
                reader, writer = await asyncio.wait_for(coro, timeout=timeout)
            else:
                reader, writer = await coro
            return _AsyncioStream(reader, writer)
        except asyncio.TimeoutError:
            raise ConnectTimeout()
        except Exception as exc:
            raise ConnectError() from exc

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


# Patch httpcore's anyio backend class to use our asyncio implementation
_anyio_mod.AnyIOBackend = _AsyncioBackend  # type: ignore[attr-defined]

# Strip newlines from API key — multi-line keys cause LocalProtocolError in HTTP/1.1
import os as _os
if 'TINKER_API_KEY' in _os.environ:
    import re as _re
    _os.environ['TINKER_API_KEY'] = _re.sub(r'\s+', '', _os.environ['TINKER_API_KEY'])

# Force HTTP/1.1 — disable HTTP/2 ALPN negotiation which Cloudflare rejects
import httpx as _httpx
_orig_async_init = _httpx.AsyncHTTPTransport.__init__
def _async_no_http2(self, *args, **kwargs):
    kwargs['http2'] = False
    _orig_async_init(self, *args, **kwargs)
_httpx.AsyncHTTPTransport.__init__ = _async_no_http2  # type: ignore[method-assign]

_orig_sync_init = _httpx.HTTPTransport.__init__
def _sync_no_http2(self, *args, **kwargs):
    kwargs['http2'] = False
    _orig_sync_init(self, *args, **kwargs)
_httpx.HTTPTransport.__init__ = _sync_no_http2  # type: ignore[method-assign]
