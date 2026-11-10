# Isolated Thread Memory

A small study-assistant scaffold demonstrating conversation persistence with LangGraph. Each conversation is identified by a thread ID: messages using the same ID continue one history, while different IDs remain isolated by the checkpointer.

## Setup

```powershell
python -m pip install -r requirements.txt
```

A Groq-compatible API key is needed only for calls to the real model client. Tests can inject a mocked client and do not need a key or network access.

## Conversation helpers

`build_thread_config(thread_id)` validates and trims a thread ID, returning the LangGraph `configurable.thread_id` configuration. `send_message(graph, thread_id, message)` validates the message, preserves its exact text, and invokes the supplied graph. `extract_reply(result)` returns the newest nonblank assistant reply as a `ConversationReply`.

Build the supplied graph with `utils.study_graph.build_study_graph`. It uses an in-memory checkpointer by default; pass a checkpointer explicitly when needed. Reuse a thread ID to continue that conversation, and use a distinct ID for each independent conversation.

## Tests

Run the test suite with:

```powershell
python -m pytest -q
```
