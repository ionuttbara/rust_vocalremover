# Vocal Remover AI 0.2.2

## Unified multiple-processing rule

All entry points now use the same `process_batch(...)` engine:

1. Load the ONNX model once.
2. Build, sort, deduplicate and count the complete supported-audio list.
3. Process the list in queue groups of at most two files (`2 + 2 + ... + 1`). The same ONNX session is reused and inference stays serialized for predictable RAM/VRAM usage.
4. Unload the model only after the complete queue has finished.

This behavior is shared by:

- Slint UI;
- file/folder drag & drop;
- Explorer Actions context menu;
- CLI/terminal invocation;
- folders and multiple paths supplied together.

## Explorer

- Added folder Actions menu support.
- Added `MultiSelectModel=Player` to the static shell verbs.
- Added a short per-action aggregation queue so Explorer invocations created for a multi-selection are merged before model loading.
- Separate queues are used for Remove vocals, Separate, and Get vocals so actions cannot mix.
- Stale Explorer request files older than five minutes are removed automatically.

## UI stability

- Processing now runs through the common batch engine.
- Individual file failures are collected and do not cancel the remainder of the queue.
- The processing worker is guarded with `catch_unwind`, and `busy` is always released when the worker returns to the UI event loop.
- Progress shows queue group, current file, successes and failures.
- Added Choose folder and folder drag & drop.
- Removed the per-chunk `Mutex<Session>`; one mutable ONNX session is retained for the complete request.

## Windows launch behavior

- Windows builds now use the GUI subsystem, so opening the app or invoking Explorer Actions does not create a Command Prompt window.
- Explicit terminal invocations attach to an existing parent console rather than creating a new one.
