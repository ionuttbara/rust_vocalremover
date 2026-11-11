# Rust Vocal Remover + Slint GUI

## Version 0.2.2

Vocal Remover AI uses one common batch-processing engine for UI, Explorer Actions and CLI.

### Batch rule

For every multiple-processing request:

1. the ONNX model is loaded once;
2. the complete input list is prepared, recursively scanning folders, then sorted and deduplicated;
3. files are arranged in queue groups of at most two (`2 + 2 + ... + 1`);
4. the same ONNX session processes the complete queue sequentially for stable RAM/VRAM usage;
5. the model is unloaded only after the complete request finishes.

A failed audio file is recorded as a failure and the remaining queue continues.

## Supported inputs

- WAV
- MP3
- M4A
- AAC
- FLAC

Folders are scanned recursively for supported audio files.

## Processing modes

- **Remove vocals** -> `<name>_instruments.wav`
- **Separate vocals and instruments** -> both stems
- **Get vocals** -> `<name>_vocals.wav`

Outputs are 32-bit float stereo WAV files.

## GUI

Run the executable without arguments. The UI supports:

- choosing multiple audio files;
- choosing a folder;
- dropping files or folders from Explorer;
- choosing an output directory;
- unified queue progress and per-file error isolation.

The Windows executable uses the GUI subsystem and does not open an extra Command Prompt window.

## Explorer Actions

Install once after upgrading:

```bat
rustvocalremover_notintegrated.exe --add
```

or, for the embedded-model build:

```bat
rustvocalremover_integrated.exe --add
```

The menu is available for supported audio files and folders:

- **Actions > Remove vocals**
- **Actions > Separate vocals and instruments**
- **Actions > Get vocals**

Multi-selection invocations created by Explorer are aggregated before the model is loaded. Aggregation is separated by action so different processing modes cannot be mixed.

Remove the menu with:

```bat
rustvocalremover_notintegrated.exe --remove-context-menu
```

## CLI

One or many files/folders can be provided in one command:

```bat
rustvocalremover_notintegrated.exe --action separate "C:\Music\song1.flac" "C:\Music\song2.m4a"
rustvocalremover_notintegrated.exe --action remove-vocals "C:\Music\Album"
rustvocalremover_notintegrated.exe --action get-vocals --output-dir "D:\Vocals" "C:\Music\Album" "C:\Music\single.wav"
```

Use `--cpu` to force CPU execution. The legacy `--threads` option is retained for command-line compatibility, but model inference deliberately reuses one session sequentially.

## Model variants

The non-integrated build reads `uvr-mdx-hq-3.onnx` from:

`HKLM\SOFTWARE\Gallery Inc\AI-ModelsManager\models`

value:

`uvr-mdx-hq-3.onnx`

The integrated build embeds `uvr-mdx-hq-3.onnx` using `include_bytes!`.

## Build

Run:

```bat
build.bat
```

The script builds `rustvocalremover_notintegrated.exe`. If the ONNX model is available locally or through AI Models Manager, it also builds `rustvocalremover_integrated.exe`.

After upgrading, reinstall Explorer Actions once so the folder and multi-select registration is refreshed.
