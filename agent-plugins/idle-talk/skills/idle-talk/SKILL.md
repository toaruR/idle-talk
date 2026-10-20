---
name: idle-talk
description: Guide a host-connected idle-talk application through one safe, user-directed idle-conversation turn.
---

# idle-talk

Use this skill only when the user asks to interact with idle-talk. Package
selection and unrelated conversations perform zero connection probes, mode-resource
loads, and application requests.

## Connection and mode

Use only the host-configured current idle-talk connection. Never start a process,
discover or select another local service, or choose an endpoint. If no current
connection is available, report `IDLE_TALK_UNAVAILABLE` and ask the user to connect
idle-talk through the host.

First call `idle_talk.status` with `{}` only after this workflow is selected. A
success is `{ "state": "ready" | "busy", "modes": string[], "connectionId": string }`.
Accept only the documented shape; any other response is
`IDLE_TALK_PROTOCOL_MISMATCH`.

Ask for a mode when the user has not stated an intent. Use `conversation` only when
the user explicitly states conversation intent. `radio` always requires an explicit
user choice. Do not issue `idle_talk.next` until the chosen mode is advertised.

## One-turn provider contract

Generate an opaque, fresh `requestId` for each user-directed turn. Allow at most one
in-flight request and keep no hidden queue.

Call `idle_talk.next` with
`{ "mode": "radio" | "conversation", "requestId": string }`. Its success is
`{ "requestId": string, "text": string, "mode": string, "cancelled": false }`.
The returned `requestId` must exactly echo the caller request ID. A missing or
mismatched request ID is `IDLE_TALK_PROTOCOL_MISMATCH`. A replay with the same ID
must return the same terminal result; otherwise report
`IDLE_TALK_PROTOCOL_MISMATCH` without starting another turn.

Call `idle_talk.cancel` only for the one running request, with
`{ "requestId": string }`. Its success is
`{ "requestId": string, "cancelled": true }`; the ID must exactly echo the caller
request ID. Send at most one cancellation request. A completion matching the ID
before cancellation is one terminal success and wins the race.

Wait while elapsed time is below 30 seconds. At or after 30 seconds, make one
cancellation attempt. Report `IDLE_TALK_TIMEOUT` only if no matching terminal result
arrives after that cancellation attempt.

## Terminal responses

Accept exactly these terminal error codes:

| Operation | Codes |
| --- | --- |
| `idle_talk.status` | `IDLE_TALK_UNAVAILABLE`, `IDLE_TALK_PROTOCOL_MISMATCH` |
| `idle_talk.next` | `IDLE_TALK_UNAVAILABLE`, `IDLE_TALK_INVALID_MODE`, `IDLE_TALK_BUSY`, `IDLE_TALK_CANCELLED`, `IDLE_TALK_TIMEOUT`, `IDLE_TALK_PROTOCOL_MISMATCH` |
| `idle_talk.cancel` | `IDLE_TALK_NOT_FOUND`, `IDLE_TALK_CANCEL_TOO_LATE`, `IDLE_TALK_UNAVAILABLE`, `IDLE_TALK_PROTOCOL_MISMATCH` |

For every terminal error, preserve the supplied `requestId`. Report only the error
code and a concise recovery action: connect idle-talk, choose an advertised mode,
wait until ready, or make a fresh user-directed request. Do not reveal endpoints,
credentials, or stack traces. Do not automatically retry, parse fallback shapes, or
make fallback calls.

## Authority boundary

This skill reads packaged instructions and uses only the named current connection's
`status`, user-directed `next`, and `cancel` operations. It grants no filesystem,
no shell authority, no arbitrary URL authority, no credential-reading authority,
and no process-management authority. It grants no filesystem authority.
