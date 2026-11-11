#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use clap::{Parser, ValueEnum};
use hound::{WavSpec, WavWriter};
use num_complex::Complex;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider},
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use rustfft::FftPlanner;
use slint::ComponentHandle;
use std::cell::RefCell;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use walkdir::WalkDir;

const STFT_CHANNELS: usize = 4;
const STFT_BINS: usize = 3072;
const STFT_FRAMES: usize = 256;
const HOP_LENGTH: usize = 1024;
const N_FFT: usize = 6144;
const CHUNK_SIZE_SAMPLES: usize = STFT_FRAMES * HOP_LENGTH;
const SUPPORTED_EXTENSIONS: [&str; 5] = ["wav", "mp3", "m4a", "aac", "flac"];
const MODEL_NAME: &str = "uvr-mdx-hq-3.onnx";

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn CreateMutexW(
        attributes: *const std::ffi::c_void,
        initial_owner: i32,
        name: *const u16,
    ) -> *mut std::ffi::c_void;
    fn WaitForSingleObject(handle: *mut std::ffi::c_void, milliseconds: u32) -> u32;
    fn ReleaseMutex(handle: *mut std::ffi::c_void) -> i32;
    fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
}

slint::slint! {
    import { Button, HorizontalBox, ProgressIndicator, RadioGroup, ScrollView, VerticalBox } from "std-widgets.slint";

    export component MainWindow inherits Window {
        title: "Vocal Remover AI";
        icon: @image-url("assets/app_icon.png");
        preferred-width: 820px;
        preferred-height: 820px;
        min-width: 720px;
        min-height: 740px;

        in-out property <int> file-count: 0;
        in-out property <string> files-text: "No files selected";
        in-out property <string> output-dir: "Same folder as input";
        in-out property <string> status-text: "Ready";
        in-out property <int> progress-value: 0;
        in-out property <bool> busy: false;
        in-out property <bool> drag-active: false;
        in-out property <int> action-mode: 1;
        in property <string> model-status: "";

        callback add-files();
        callback add-folder();
        callback clear-files();
        callback choose-output();
        callback start-processing();
        callback install-context-menu();

        background: rgb(21, 23, 27);

        VerticalBox {
            padding: 22px;
            spacing: 14px;

            HorizontalBox {
                spacing: 14px;
                Image {
                    source: @image-url("assets/app_icon.png");
                    width: 56px;
                    height: 56px;
                    image-fit: contain;
                }
                VerticalBox {
                    spacing: 3px;
                    Text {
                        text: "AI Vocal Remover";
                        color: rgb(245, 247, 250);
                        font-size: 28px;
                        font-weight: 700;
                        vertical-alignment: center;
                    }
                    Text {
                        text: root.model-status;
                        color: rgb(174, 182, 194);
                        font-size: 13px;
                        vertical-alignment: center;
                    }
                }
            }

            drop-zone := Rectangle {
                height: 200px;
                border-radius: 14px;
                border-width: 2px;
                border-color: root.drag-active ? rgb(102, 167, 255) : rgb(66, 73, 85);
                background: root.drag-active ? rgb(30, 43, 63) : rgb(28, 32, 38);

                VerticalBox {
                    padding: 22px;
                    spacing: 8px;
                    alignment: center;

                    Text {
                        text: root.drag-active ? "Drop audio file or folder here" : "Drag & drop audio or folders here";
                        color: rgb(244, 246, 248);
                        font-size: 20px;
                        horizontal-alignment: center;
                    }
                    Text {
                        text: "WAV  •  MP3  •  M4A  •  AAC  •  FLAC";
                        color: rgb(155, 165, 178);
                        font-size: 13px;
                        horizontal-alignment: center;
                    }
                    HorizontalBox {
                        spacing: 10px;
                        alignment: center;
                        Button {
                            text: "Choose audio files";
                            enabled: !root.busy;
                            clicked => { root.add-files(); }
                        }
                        Button {
                            text: "Choose folder";
                            enabled: !root.busy;
                            clicked => { root.add-folder(); }
                        }
                    }
                }
            }

            HorizontalBox {
                spacing: 10px;
                Text {
                    text: root.file-count + " input(s)";
                    color: rgb(232, 235, 239);
                    vertical-alignment: center;
                }
                Rectangle { horizontal-stretch: 1; }
                Button {
                    text: "Clear";
                    enabled: !root.busy && root.file-count > 0;
                    clicked => { root.clear-files(); }
                }
            }

            ScrollView {
                height: 135px;
                Text {
                    text: root.files-text;
                    color: rgb(198, 204, 213);
                    font-size: 12px;
                    wrap: word-wrap;
                }
            }

            Rectangle {
                height: 48px;
                border-radius: 8px;
                background: rgb(28, 32, 38);
                HorizontalBox {
                    padding: 10px;
                    spacing: 10px;
                    Text {
                        text: "Save to:";
                        color: rgb(220, 225, 231);
                        vertical-alignment: center;
                    }
                    Text {
                        text: root.output-dir;
                        color: rgb(174, 182, 194);
                        overflow: elide;
                        vertical-alignment: center;
                        horizontal-stretch: 1;
                    }
                    Button {
                        text: "Choose folder";
                        enabled: !root.busy;
                        clicked => { root.choose-output(); }
                    }
                }
            }

            Text {
                text: "Action";
                color: rgb(232, 235, 239);
                font-size: 14px;
                font-weight: 600;
            }

            RadioGroup {
                orientation: horizontal;
                enabled: !root.busy;
                RadioButton { text: "Remove vocals"; }
                RadioButton { text: "Separate vocals + instruments"; checked: true; }
                RadioButton { text: "Get vocals"; }
                selected(value) => {
                    if value == "Remove vocals" {
                        root.action-mode = 0;
                    } else if value == "Get vocals" {
                        root.action-mode = 2;
                    } else {
                        root.action-mode = 1;
                    }
                }
            }

            ProgressIndicator {
                progress: root.progress-value / 100.0;
            }

            Text {
                text: root.status-text;
                color: root.busy ? rgb(142, 192, 255) : rgb(174, 182, 194);
                font-size: 13px;
            }

            HorizontalBox {
                spacing: 10px;
                Button {
                    text: "Install Explorer Actions menu";
                    enabled: !root.busy;
                    clicked => { root.install-context-menu(); }
                }
                Rectangle { horizontal-stretch: 1; }
                Button {
                    text: root.busy ? "Processing..." : "Process and save";
                    enabled: !root.busy && root.file-count > 0;
                    clicked => { root.start-processing(); }
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum SeparationAction {
    #[value(name = "remove-vocals")]
    RemoveVocals,
    #[value(name = "separate")]
    Separate,
    #[value(name = "get-vocals")]
    GetVocals,
}

impl SeparationAction {
    fn from_ui(value: i32) -> Self {
        match value {
            0 => Self::RemoveVocals,
            2 => Self::GetVocals,
            _ => Self::Separate,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::RemoveVocals => "Remove vocals",
            Self::Separate => "Separate vocals + instruments",
            Self::GetVocals => "Get vocals",
        }
    }

    fn queue_key(self) -> &'static str {
        match self {
            Self::RemoveVocals => "remove-vocals",
            Self::Separate => "separate",
            Self::GetVocals => "get-vocals",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "AI vocal remover with Slint GUI and Windows Explorer actions")]
struct Args {
    /// File(s) or folder(s) to process. If omitted, the Slint GUI is opened.
    #[arg(index = 1, value_name = "INPUT", num_args = 0..)]
    input: Vec<PathBuf>,

    /// Install Actions > Remove vocals / Separate / Get vocals in Explorer.
    #[arg(long)]
    add: bool,

    /// Remove the Explorer Actions menu entries installed by this app.
    #[arg(long)]
    remove_context_menu: bool,

    /// Processing action used for command-line/context-menu invocation.
    #[arg(long, value_enum, default_value_t = SeparationAction::Separate)]
    action: SeparationAction,

    /// Internal flag used by Explorer Actions. Shows a dialog if processing fails.
    #[arg(long, hide = true)]
    context_menu: bool,

    /// Optional output directory. By default outputs are written next to each input file.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Force CPU execution.
    #[arg(long)]
    cpu: bool,

    /// Retained for command-line compatibility. Multi-audio processing uses one model session and queues files in pairs.
    #[arg(short = 't', long, default_value_t = 1)]
    threads: usize,
}

#[cfg(target_os = "windows")]
fn configure_process_priority() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, SetPriorityClass, IDLE_PRIORITY_CLASS,
    };
    unsafe {
        let handle = GetCurrentProcess();
        let _ = SetPriorityClass(handle, IDLE_PRIORITY_CLASS);
    }
}

#[cfg(not(target_os = "windows"))]
fn configure_process_priority() {}

#[cfg(target_os = "windows")]
fn attach_parent_console() -> bool {
    use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    unsafe { AttachConsole(ATTACH_PARENT_PROCESS) != 0 }
}

#[cfg(not(target_os = "windows"))]
fn attach_parent_console() -> bool {
    true
}

fn is_supported_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn collect_audio_files(input: &Path) -> Vec<PathBuf> {
    if input.is_file() {
        return if is_supported_audio(input) {
            vec![input.to_path_buf()]
        } else {
            Vec::new()
        };
    }

    if input.is_dir() {
        return WalkDir::new(input)
            .into_iter()
            .filter_map(Result::ok)
            .map(|entry| entry.into_path())
            .filter(|path| path.is_file() && is_supported_audio(path))
            .collect();
    }

    Vec::new()
}

fn add_path_to_list(files: &mut Vec<PathBuf>, path: &Path) {
    for candidate in collect_audio_files(path) {
        if !files.contains(&candidate) {
            files.push(candidate);
        }
    }
    files.sort();
}

fn prepare_input_list(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for input in inputs {
        add_path_to_list(&mut files, input);
    }
    files.sort();
    files.dedup();
    files
}

fn format_file_list(files: &[PathBuf]) -> String {
    if files.is_empty() {
        return "No files selected".to_string();
    }

    files
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn sync_gui_file_state(ui: &MainWindow, files: &[PathBuf]) {
    ui.set_file_count(files.len() as i32);
    ui.set_files_text(format_file_list(files).into());
}

#[cfg(feature = "integrated")]
fn model_status_text() -> String {
    format!("AI model: {} embedded in this executable", MODEL_NAME)
}

#[cfg(all(not(feature = "integrated"), target_os = "windows"))]
fn model_status_text() -> String {
    match ai_manager_model_path() {
        Ok(path) if path.exists() => format!("AI Models Manager: {}", path.display()),
        Ok(path) => format!("AI Models Manager entry found, but file is missing: {}", path.display()),
        Err(_) => format!("AI Models Manager: {} not registered", MODEL_NAME),
    }
}

#[cfg(all(not(feature = "integrated"), not(target_os = "windows")))]
fn model_status_text() -> String {
    format!("External model required: set VOCAL_REMOVER_MODEL to {}", MODEL_NAME)
}

#[cfg(target_os = "windows")]
fn ai_manager_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    use winreg::enums::*;
    use winreg::RegKey;

    // Preserve the AI Models Manager location used by the original project.
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let key = hklm.open_subkey(r#"SOFTWARE\Gallery Inc\AI-ModelsManager\models"#)?;
    let value: String = key.get_value(MODEL_NAME)?;
    Ok(PathBuf::from(value))
}

#[cfg(all(not(feature = "integrated"), not(target_os = "windows")))]
fn external_model_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let value = std::env::var("VOCAL_REMOVER_MODEL")?;
    Ok(PathBuf::from(value))
}

fn create_session(force_cpu: bool) -> Result<Session, Box<dyn std::error::Error>> {
    let _ = ort::init().with_name("rustvocalremover").commit();

    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(2)?;

    if force_cpu {
        builder = builder.with_execution_providers([CPUExecutionProvider::default().build()])?;
    } else {
        builder = builder.with_execution_providers([
            CUDAExecutionProvider::default().build(),
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?;
    }

    #[cfg(feature = "integrated")]
    {
        const MODEL_BYTES: &[u8] = include_bytes!("../uvr-mdx-hq-3.onnx");
        return Ok(builder.commit_from_memory(MODEL_BYTES)?);
    }

    #[cfg(all(not(feature = "integrated"), target_os = "windows"))]
    {
        let model_path = ai_manager_model_path()?;
        if !model_path.exists() {
            return Err(format!("AI model does not exist at: {}", model_path.display()).into());
        }
        return Ok(builder.commit_from_file(model_path)?);
    }

    #[cfg(all(not(feature = "integrated"), not(target_os = "windows")))]
    {
        let model_path = external_model_path()?;
        if !model_path.exists() {
            return Err(format!("AI model does not exist at: {}", model_path.display()).into());
        }
        Ok(builder.commit_from_file(model_path)?)
    }
}

#[cfg(target_os = "windows")]
fn register_context_menu() -> Result<(), Box<dyn std::error::Error>> {
    use winreg::enums::*;
    use winreg::RegKey;

    fn install_actions(
        hkcu: &RegKey,
        base: &str,
        exe_path: &str,
        exe_icon: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _ = hkcu.delete_subkey_all(base);

        let (actions_key, _) = hkcu.create_subkey(base)?;
        actions_key.set_value("MUIVerb", &"Actions")?;
        actions_key.set_value("Icon", &exe_icon)?;
        actions_key.set_value("SubCommands", &"")?;
        actions_key.set_value("MultiSelectModel", &"Player")?;

        let commands = [
            ("01_RemoveVocals", "Remove vocals", "remove-vocals"),
            ("02_Separate", "Separate vocals and instruments", "separate"),
            ("03_GetVocals", "Get vocals", "get-vocals"),
        ];

        for (key_name, menu_text, action) in commands {
            let sub_path = format!(r#"{}\shell\{}"#, base, key_name);
            let (sub_key, _) = hkcu.create_subkey(&sub_path)?;
            sub_key.set_value("", &menu_text)?;
            sub_key.set_value("MUIVerb", &menu_text)?;
            sub_key.set_value("Icon", &exe_icon)?;
            sub_key.set_value("MultiSelectModel", &"Player")?;

            let (command_key, _) = sub_key.create_subkey("command")?;
            command_key.set_value(
                "",
                &format!(
                    "\"{}\" --context-menu --action {} -- \"%1\"",
                    exe_path, action
                ),
            )?;
        }
        Ok(())
    }

    let exe_path = std::env::current_exe()?.display().to_string();
    let exe_icon = format!("{},0", exe_path);
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);

    for ext in [".wav", ".mp3", ".m4a", ".aac", ".flac"] {
        let legacy = format!(
            r#"Software\Classes\SystemFileAssociations\{}\shell\RemoveVocals"#,
            ext
        );
        let base = format!(
            r#"Software\Classes\SystemFileAssociations\{}\shell\VocalRemoverActions"#,
            ext
        );
        let _ = hkcu.delete_subkey_all(&legacy);
        install_actions(&hkcu, &base, &exe_path, &exe_icon)?;
    }

    // Folder support follows the same unified batch engine as UI and CLI.
    install_actions(
        &hkcu,
        r#"Software\Classes\Directory\shell\VocalRemoverActions"#,
        &exe_path,
        &exe_icon,
    )?;

    notify_shell_association_changed();
    Ok(())
}

#[cfg(target_os = "windows")]
fn notify_shell_association_changed() {
    use std::ffi::c_void;

    const SHCNE_ASSOCCHANGED: i32 = 0x0800_0000;
    const SHCNF_IDLIST: u32 = 0x0000;

    #[link(name = "shell32")]
    extern "system" {
        fn SHChangeNotify(
            w_event_id: i32,
            u_flags: u32,
            item1: *const c_void,
            item2: *const c_void,
        );
    }

    unsafe {
        SHChangeNotify(
            SHCNE_ASSOCCHANGED,
            SHCNF_IDLIST,
            std::ptr::null(),
            std::ptr::null(),
        );
    }
}

#[cfg(not(target_os = "windows"))]
fn register_context_menu() -> Result<(), Box<dyn std::error::Error>> {
    Err("Explorer context menu registration is only available on Windows".into())
}

#[cfg(target_os = "windows")]
fn unregister_context_menu() -> Result<(), Box<dyn std::error::Error>> {
    use winreg::enums::*;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    for ext in [".wav", ".mp3", ".m4a", ".aac", ".flac"] {
        let path = format!(
            r#"Software\Classes\SystemFileAssociations\{}\shell\VocalRemoverActions"#,
            ext
        );
        let legacy = format!(
            r#"Software\Classes\SystemFileAssociations\{}\shell\RemoveVocals"#,
            ext
        );
        let _ = hkcu.delete_subkey_all(path);
        let _ = hkcu.delete_subkey_all(legacy);
    }
    let _ = hkcu.delete_subkey_all(r#"Software\Classes\Directory\shell\VocalRemoverActions"#);
    notify_shell_association_changed();
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn unregister_context_menu() -> Result<(), Box<dyn std::error::Error>> {
    Err("Explorer context menu registration is only available on Windows".into())
}

fn output_directory_for(input: &Path, chosen: Option<&Path>) -> PathBuf {
    chosen
        .map(Path::to_path_buf)
        .unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")).to_path_buf())
}

#[derive(Debug, Clone)]
struct BatchFailure {
    path: PathBuf,
    error: String,
}

#[derive(Debug, Clone, Default)]
struct BatchSummary {
    total: usize,
    succeeded: usize,
    failures: Vec<BatchFailure>,
}

#[derive(Debug, Clone)]
enum ProcessingEvent {
    LoadingModel,
    ListReady {
        total: usize,
        batches: usize,
    },
    FileStarted {
        index: usize,
        total: usize,
        batch: usize,
        batches: usize,
        path: PathBuf,
    },
    FileFinished {
        index: usize,
        total: usize,
        succeeded: usize,
        failed: usize,
    },
    UnloadingModel,
}

fn process_batch<F>(
    inputs: &[PathBuf],
    chosen_output: Option<&Path>,
    force_cpu: bool,
    action: SeparationAction,
    mut on_event: F,
) -> Result<BatchSummary, Box<dyn std::error::Error>>
where
    F: FnMut(ProcessingEvent),
{
    // Rule 1: one model lifecycle for the complete request, regardless of caller.
    on_event(ProcessingEvent::LoadingModel);
    let mut session = create_session(force_cpu)?;

    // Rule 2: prepare, normalize and count the complete input list only after model startup.
    let files = prepare_input_list(inputs);
    if files.is_empty() {
        on_event(ProcessingEvent::UnloadingModel);
        drop(session);
        return Err("No supported audio files found".into());
    }

    let input_name = session
        .inputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or("The ONNX model does not report an input")?;

    let total = files.len();
    let batches = (total + 1) / 2;
    on_event(ProcessingEvent::ListReady { total, batches });

    let mut summary = BatchSummary {
        total,
        ..BatchSummary::default()
    };
    let mut completed = 0usize;

    // Rule 3: queue groups contain at most two files. Inference itself stays
    // serialized through the same Session for predictable RAM/VRAM usage.
    for (batch_index, pair) in files.chunks(2).enumerate() {
        for file in pair {
            let index = completed + 1;
            on_event(ProcessingEvent::FileStarted {
                index,
                total,
                batch: batch_index + 1,
                batches,
                path: file.clone(),
            });

            let result = (|| -> Result<(), Box<dyn std::error::Error>> {
                let output_dir = output_directory_for(file, chosen_output);
                std::fs::create_dir_all(&output_dir)?;
                process_audio(file, &output_dir, &mut session, &input_name, action)
            })();

            match result {
                Ok(()) => summary.succeeded += 1,
                Err(err) => summary.failures.push(BatchFailure {
                    path: file.clone(),
                    error: err.to_string(),
                }),
            }

            completed += 1;
            on_event(ProcessingEvent::FileFinished {
                index: completed,
                total,
                succeeded: summary.succeeded,
                failed: summary.failures.len(),
            });
        }
    }

    // Rule 4: unload only after the complete queue is finished.
    on_event(ProcessingEvent::UnloadingModel);
    drop(session);
    Ok(summary)
}

fn run_cli(
    args: &Args,
    inputs: &[PathBuf],
    verbose: bool,
) -> Result<BatchSummary, Box<dyn std::error::Error>> {
    let _requested_threads = args.threads.max(1);

    let summary = process_batch(
        inputs,
        args.output_dir.as_deref(),
        args.cpu,
        args.action,
        |event| {
            if !verbose {
                return;
            }
            match event {
                ProcessingEvent::LoadingModel => println!("Loading AI model..."),
                ProcessingEvent::ListReady { total, batches } => {
                    println!(
                        "Prepared {total} audio file(s) in {batches} queue group(s), maximum 2 per group."
                    );
                }
                ProcessingEvent::FileStarted {
                    index,
                    total,
                    batch,
                    batches,
                    path,
                } => {
                    println!(
                        "Processing {index}/{total} (queue {batch}/{batches}): {}",
                        path.display()
                    );
                }
                ProcessingEvent::FileFinished {
                    index,
                    total,
                    succeeded,
                    failed,
                } => {
                    println!("Progress {index}/{total} - OK: {succeeded}, failed: {failed}");
                }
                ProcessingEvent::UnloadingModel => println!("Unloading AI model..."),
            }
        },
    )?;

    if verbose {
        println!(
            "Done. {} succeeded, {} failed.",
            summary.succeeded,
            summary.failures.len()
        );
        for failure in &summary.failures {
            eprintln!("Failed: {} - {}", failure.path.display(), failure.error);
        }
    }

    Ok(summary)
}

fn run_gui(force_cpu: bool) -> Result<(), Box<dyn std::error::Error>> {
    use slint::winit_030::{winit, EventResult, WinitWindowAccessor};

    slint::BackendSelector::new()
        .backend_name("winit".into())
        .select()?;

    let ui = MainWindow::new()?;
    ui.set_model_status(model_status_text().into());

    let files_state: Rc<RefCell<Vec<PathBuf>>> = Rc::new(RefCell::new(Vec::new()));
    let output_state: Rc<RefCell<Option<PathBuf>>> = Rc::new(RefCell::new(None));

    {
        let weak = ui.as_weak();
        let files_state = files_state.clone();
        ui.on_add_files(move || {
            let Some(paths) = rfd::FileDialog::new()
                .add_filter("Audio", &["wav", "mp3", "m4a", "aac", "flac"])
                .pick_files()
            else {
                return;
            };

            let mut files = files_state.borrow_mut();
            for path in paths {
                if is_supported_audio(&path) && !files.contains(&path) {
                    files.push(path);
                }
            }
            files.sort();
            if let Some(ui) = weak.upgrade() {
                sync_gui_file_state(&ui, &files);
                ui.set_status_text("Ready".into());
            }
        });
    }

    {
        let weak = ui.as_weak();
        let files_state = files_state.clone();
        ui.on_add_folder(move || {
            let Some(folder) = rfd::FileDialog::new().pick_folder() else {
                return;
            };
            let mut files = files_state.borrow_mut();
            if !files.contains(&folder) {
                files.push(folder);
                files.sort();
            }
            if let Some(ui) = weak.upgrade() {
                sync_gui_file_state(&ui, &files);
                ui.set_status_text("Folder added. It will be scanned after the AI model starts.".into());
            }
        });
    }

    {
        let weak = ui.as_weak();
        let files_state = files_state.clone();
        ui.on_clear_files(move || {
            files_state.borrow_mut().clear();
            if let Some(ui) = weak.upgrade() {
                sync_gui_file_state(&ui, &[]);
                ui.set_progress_value(0);
                ui.set_status_text("Ready".into());
            }
        });
    }

    {
        let weak = ui.as_weak();
        let output_state = output_state.clone();
        ui.on_choose_output(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                *output_state.borrow_mut() = Some(folder.clone());
                if let Some(ui) = weak.upgrade() {
                    ui.set_output_dir(folder.display().to_string().into());
                }
            }
        });
    }

    {
        let weak = ui.as_weak();
        ui.on_install_context_menu(move || {
            let result = register_context_menu();
            if let Some(ui) = weak.upgrade() {
                match result {
                    Ok(()) => ui.set_status_text(
                        "Explorer menu installed for supported audio files and folders.".into(),
                    ),
                    Err(err) => ui.set_status_text(format!("Context menu error: {err}").into()),
                }
            }
        });
    }

    {
        let weak = ui.as_weak();
        let files_state = files_state.clone();
        let output_state = output_state.clone();
        ui.on_start_processing(move || {
            let Some(ui_now) = weak.upgrade() else {
                return;
            };
            if ui_now.get_busy() {
                return;
            }

            let files = files_state.borrow().clone();
            if files.is_empty() {
                ui_now.set_status_text("Add at least one supported audio file.".into());
                return;
            }

            let output_dir = output_state.borrow().clone();
            let action = SeparationAction::from_ui(ui_now.get_action_mode());
            ui_now.set_busy(true);
            ui_now.set_progress_value(0);
            ui_now.set_status_text(format!("Loading AI model… {}", action.label()).into());

            let weak_worker = weak.clone();
            std::thread::spawn(move || {
                let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    process_batch(
                        &files,
                        output_dir.as_deref(),
                        force_cpu,
                        action,
                        |event| {
                            let weak_progress = weak_worker.clone();
                            match event {
                                ProcessingEvent::LoadingModel => {
                                    weak_progress
                                        .upgrade_in_event_loop(move |ui| {
                                            ui.set_status_text(
                                                format!("Loading AI model… {}", action.label()).into(),
                                            );
                                        })
                                        .ok();
                                }
                                ProcessingEvent::ListReady { total, batches } => {
                                    weak_progress
                                        .upgrade_in_event_loop(move |ui| {
                                            ui.set_status_text(
                                                format!(
                                                    "Prepared {total} audio file(s) in {batches} queue group(s)."
                                                )
                                                .into(),
                                            );
                                        })
                                        .ok();
                                }
                                ProcessingEvent::FileStarted {
                                    index,
                                    total,
                                    batch,
                                    batches,
                                    path,
                                } => {
                                    let display_name = path
                                        .file_name()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("audio")
                                        .to_string();
                                    let progress_before =
                                        (((index.saturating_sub(1)) * 100) / total.max(1)) as i32;
                                    weak_progress
                                        .upgrade_in_event_loop(move |ui| {
                                            ui.set_progress_value(progress_before);
                                            ui.set_status_text(
                                                format!(
                                                    "Queue {batch}/{batches} · {index}/{total}: {display_name}"
                                                )
                                                .into(),
                                            );
                                        })
                                        .ok();
                                }
                                ProcessingEvent::FileFinished {
                                    index,
                                    total,
                                    succeeded,
                                    failed,
                                } => {
                                    let progress = ((index * 100) / total.max(1)) as i32;
                                    weak_progress
                                        .upgrade_in_event_loop(move |ui| {
                                            ui.set_progress_value(progress);
                                            ui.set_status_text(
                                                format!(
                                                    "Processed {index}/{total} · OK {succeeded} · Failed {failed}"
                                                )
                                                .into(),
                                            );
                                        })
                                        .ok();
                                }
                                ProcessingEvent::UnloadingModel => {
                                    weak_progress
                                        .upgrade_in_event_loop(|ui| {
                                            ui.set_status_text(
                                                "Finishing queue and unloading AI model...".into(),
                                            );
                                        })
                                        .ok();
                                }
                            }
                        },
                    )
                    .map_err(|e| e.to_string())
                }));

                let result: Result<BatchSummary, String> = match worker_result {
                    Ok(result) => result,
                    Err(_) => Err(
                        "The processing worker stopped unexpectedly. The UI remains usable; try the batch again."
                            .to_string(),
                    ),
                };

                let weak_done = weak_worker.clone();
                weak_done
                    .upgrade_in_event_loop(move |ui| {
                        ui.set_busy(false);
                        match result {
                            Ok(summary) if summary.failures.is_empty() => {
                                ui.set_progress_value(100);
                                ui.set_status_text(
                                    format!(
                                        "Finished. {} audio file(s) processed.",
                                        summary.succeeded
                                    )
                                    .into(),
                                );
                            }
                            Ok(summary) => {
                                ui.set_progress_value(100);
                                let first = summary
                                    .failures
                                    .first()
                                    .map(|f| format!(" First error: {}", f.error))
                                    .unwrap_or_default();
                                ui.set_status_text(
                                    format!(
                                        "Finished: {} succeeded, {} failed.{}",
                                        summary.succeeded,
                                        summary.failures.len(),
                                        first
                                    )
                                    .into(),
                                );
                            }
                            Err(err) => {
                                ui.set_status_text(format!("Error: {err}").into());
                            }
                        }
                    })
                    .ok();
            });
        });
    }

    {
        let weak = ui.as_weak();
        let files_state = files_state.clone();
        ui.window().on_winit_window_event(move |_window, event| {
            match event {
                winit::event::WindowEvent::HoveredFile(path) => {
                    if let Some(ui) = weak.upgrade() {
                        ui.set_drag_active(path.is_dir() || is_supported_audio(&path));
                    }
                }
                winit::event::WindowEvent::HoveredFileCancelled => {
                    if let Some(ui) = weak.upgrade() {
                        ui.set_drag_active(false);
                    }
                }
                winit::event::WindowEvent::DroppedFile(path) => {
                    if let Some(ui) = weak.upgrade() {
                        ui.set_drag_active(false);
                    }

                    if path.is_dir() || is_supported_audio(&path) {
                        let mut files = files_state.borrow_mut();
                        if !files.contains(&path) {
                            files.push(path.clone());
                            files.sort();
                        }
                        if let Some(ui) = weak.upgrade() {
                            sync_gui_file_state(&ui, &files);
                            ui.set_status_text(
                                if path.is_dir() {
                                    "Folder added. It will be scanned after the AI model starts.".into()
                                } else {
                                    "Audio file added. Choose an action and press Process and save.".into()
                                },
                            );
                        }
                    } else if let Some(ui) = weak.upgrade() {
                        ui.set_status_text(
                            "Unsupported file. Use WAV, MP3, M4A, AAC or FLAC.".into(),
                        );
                    }
                }
                _ => {}
            }
            EventResult::Propagate
        });
    }

    ui.run()?;
    Ok(())
}

#[cfg(target_os = "windows")]
struct ContextBatchLock(*mut std::ffi::c_void);

#[cfg(target_os = "windows")]
impl Drop for ContextBatchLock {
    fn drop(&mut self) {
        unsafe {
            let _ = ReleaseMutex(self.0);
            let _ = CloseHandle(self.0);
        }
    }
}

struct ContextBatchLease {
    inputs: Vec<PathBuf>,
    #[cfg(target_os = "windows")]
    _lock: ContextBatchLock,
}

#[cfg(target_os = "windows")]
fn context_queue_dir(action: SeparationAction) -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("Gallery Inc")
        .join("Vocal Remover AI")
        .join("context-queue")
        .join(action.queue_key())
}

#[cfg(target_os = "windows")]
fn cleanup_stale_context_requests(action: SeparationAction) {
    use std::time::{Duration, SystemTime};

    let Ok(entries) = std::fs::read_dir(context_queue_dir(action)) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !matches!(path.extension().and_then(|s| s.to_str()), Some("req" | "tmp")) {
            continue;
        }
        let stale = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .map(|age| age > Duration::from_secs(300))
            .unwrap_or(false);
        if stale {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(target_os = "windows")]
fn enqueue_context_inputs(
    inputs: &[PathBuf],
    action: SeparationAction,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::windows::ffi::OsStrExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    let dir = context_queue_dir(action);
    std::fs::create_dir_all(&dir)?;
    cleanup_stale_context_requests(action);
    let pid = std::process::id();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

    for (index, input) in inputs.iter().enumerate() {
        let request_path = dir.join(format!("{pid}-{stamp}-{index}.req"));
        let temp_path = dir.join(format!("{pid}-{stamp}-{index}.tmp"));
        let mut bytes = Vec::new();
        for unit in input.as_os_str().encode_wide() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        std::fs::write(&temp_path, bytes)?;
        std::fs::rename(&temp_path, &request_path)?;
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn acquire_context_batch_lock(
    action: SeparationAction,
) -> Result<Option<ContextBatchLock>, Box<dyn std::error::Error>> {
    use std::os::windows::ffi::OsStrExt;

    const WAIT_OBJECT_0: u32 = 0;
    const WAIT_TIMEOUT: u32 = 258;

    let mutex_name = format!(
        r"Local\GalleryInc.VocalRemoverAI.ContextBatch.{}",
        action.queue_key()
    );
    let name: Vec<u16> = std::ffi::OsStr::new(&mutex_name)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error().into());
    }

    match unsafe { WaitForSingleObject(handle, 0) } {
        WAIT_OBJECT_0 => Ok(Some(ContextBatchLock(handle))),
        WAIT_TIMEOUT => {
            unsafe {
                let _ = CloseHandle(handle);
            }
            Ok(None)
        }
        _ => {
            unsafe {
                let _ = CloseHandle(handle);
            }
            Err(std::io::Error::last_os_error().into())
        }
    }
}

#[cfg(target_os = "windows")]
fn read_context_request(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use std::os::windows::ffi::OsStringExt;

    let bytes = std::fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err("Invalid Explorer batch request".into());
    }
    let mut wide = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        wide.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(PathBuf::from(std::ffi::OsString::from_wide(&wide)))
}

#[cfg(target_os = "windows")]
fn pending_context_request_count(action: SeparationAction) -> usize {
    let Ok(entries) = std::fs::read_dir(context_queue_dir(action)) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|s| s.to_str()) == Some("req"))
        .count()
}

#[cfg(target_os = "windows")]
fn drain_context_requests(action: SeparationAction) -> Vec<PathBuf> {
    let dir = context_queue_dir(action);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut requests = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("req") {
            continue;
        }
        if let Ok(input) = read_context_request(&path) {
            requests.push(input);
        }
        let _ = std::fs::remove_file(path);
    }
    requests.sort();
    requests.dedup();
    requests
}

#[cfg(target_os = "windows")]
fn collect_context_batch(
    inputs: &[PathBuf],
    action: SeparationAction,
) -> Result<Option<ContextBatchLease>, Box<dyn std::error::Error>> {
    use std::time::{Duration, Instant};

    enqueue_context_inputs(inputs, action)?;
    let Some(lock) = acquire_context_batch_lock(action)? else {
        // Another instance owns this action's queue and will collect this request.
        return Ok(None);
    };

    // Explorer can start one process per selected file. Wait for a short quiet
    // period so all invocations collapse into one list before the model starts.
    let started = Instant::now();
    let mut last_count = pending_context_request_count(action);
    let mut stable_since = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(75));
        let count = pending_context_request_count(action);
        if count != last_count {
            last_count = count;
            stable_since = Instant::now();
        }
        if stable_since.elapsed() >= Duration::from_millis(400)
            || started.elapsed() >= Duration::from_millis(1800)
        {
            break;
        }
    }

    let collected = drain_context_requests(action);
    Ok(Some(ContextBatchLease {
        inputs: collected,
        _lock: lock,
    }))
}

#[cfg(not(target_os = "windows"))]
fn collect_context_batch(
    inputs: &[PathBuf],
    _action: SeparationAction,
) -> Result<Option<ContextBatchLease>, Box<dyn std::error::Error>> {
    Ok(Some(ContextBatchLease {
        inputs: inputs.to_vec(),
    }))
}

fn show_context_summary(summary: &BatchSummary) {
    if summary.failures.is_empty() {
        return;
    }

    let mut details = format!(
        "Processed {} audio file(s): {} succeeded, {} failed.",
        summary.total,
        summary.succeeded,
        summary.failures.len()
    );
    for failure in summary.failures.iter().take(5) {
        details.push_str(&format!(
            "\n\n{}\n{}",
            failure.path.display(),
            failure.error
        ));
    }
    if summary.failures.len() > 5 {
        details.push_str("\n\nAdditional failed audio files were omitted from this dialog.");
    }

    let _ = rfd::MessageDialog::new()
        .set_title("Vocal Remover AI")
        .set_description(details)
        .set_level(rfd::MessageLevel::Warning)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // GUI/context-menu launches never create a console window. Explicit CLI
    // invocations attach to an already existing terminal instead.
    let terminal_mode = !args.context_menu
        && (args.add || args.remove_context_menu || !args.input.is_empty());
    let has_console = if terminal_mode {
        attach_parent_console()
    } else {
        false
    };

    if !args.input.is_empty() {
        configure_process_priority();
    }

    if args.add {
        register_context_menu()?;
        if has_console {
            println!("Explorer Actions menu installed for supported audio files and folders.");
        }
        return Ok(());
    }

    if args.remove_context_menu {
        unregister_context_menu()?;
        if has_console {
            println!("Explorer Actions menu removed.");
        }
        return Ok(());
    }

    if args.context_menu {
        if args.input.is_empty() {
            return Ok(());
        }

        let Some(lease) = collect_context_batch(&args.input, args.action)? else {
            return Ok(());
        };

        if !lease.inputs.is_empty() {
            match run_cli(&args, &lease.inputs, false) {
                Ok(summary) => show_context_summary(&summary),
                Err(err) => {
                    let _ = rfd::MessageDialog::new()
                        .set_title("Vocal Remover AI")
                        .set_description(format!("Explorer Action failed:\n\n{err}"))
                        .set_level(rfd::MessageLevel::Error)
                        .set_buttons(rfd::MessageButtons::Ok)
                        .show();
                }
            }
        }
        return Ok(());
    }

    if !args.input.is_empty() {
        let summary = run_cli(&args, &args.input, has_console)?;
        if summary.succeeded == 0 && !summary.failures.is_empty() {
            return Err("All audio files in the batch failed".into());
        }
        return Ok(());
    }

    run_gui(args.cpu)
}

fn process_audio(
    input_path: &Path,
    output_dir: &Path,
    session: &mut Session,
    input_name: &str,
    action: SeparationAction,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = std::fs::File::open(input_path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = input_path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("No default audio track found")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);

    let mut left_channel = Vec::new();
    let mut right_channel = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(e.into()),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };

        sample_rate = decoded.spec().rate;
        let channels = decoded.spec().channels.count().max(1);
        let mut sample_buf =
            SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        for frame in samples.chunks(channels) {
            let left = frame.first().copied().unwrap_or(0.0);
            let right = frame.get(1).copied().unwrap_or(left);
            left_channel.push(left);
            right_channel.push(right);
        }
    }

    let num_samples = left_channel.len();
    if num_samples == 0 {
        return Err("Decoded audio contains no samples".into());
    }

    let mut final_vocals_left = vec![0f32; num_samples];
    let mut final_vocals_right = vec![0f32; num_samples];
    let mut final_inst_left = vec![0f32; num_samples];
    let mut final_inst_right = vec![0f32; num_samples];

    let mut offset = 0;
    while offset < num_samples {
        let end = (offset + CHUNK_SIZE_SAMPLES).min(num_samples);
        let chunk_left = &left_channel[offset..end];
        let chunk_right = &right_channel[offset..end];

        let stft_tensor = perform_stft(chunk_left, chunk_right);
        let stft_clone = stft_tensor.clone();

        let shape = [
            1i64,
            STFT_CHANNELS as i64,
            STFT_BINS as i64,
            STFT_FRAMES as i64,
        ];
        let input_val = Value::from_array((shape, stft_tensor))?;

        let (vocals_stft, inst_stft) = {
            let outputs = session.run(inputs![input_name => input_val])?;

            if outputs.len() == 2 {
                let v_out = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
                let i_out = outputs[1].try_extract_tensor::<f32>()?.1.to_vec();
                (v_out, i_out)
            } else if outputs.len() == 1 {
                let i_out = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
                let mut v_out = vec![0.0f32; i_out.len()];
                for (j, &mix_val) in stft_clone.iter().enumerate() {
                    if j < i_out.len() {
                        v_out[j] = mix_val - i_out[j];
                    }
                }
                (v_out, i_out)
            } else {
                return Err(format!(
                    "Unexpected number of model outputs: {}",
                    outputs.len()
                )
                .into());
            }
        };

        let expected_len = STFT_CHANNELS * STFT_BINS * STFT_FRAMES;
        if vocals_stft.len() < expected_len || inst_stft.len() < expected_len {
            return Err(format!(
                "Unexpected model tensor size. vocals={}, instruments={}, expected at least {}",
                vocals_stft.len(),
                inst_stft.len(),
                expected_len
            )
            .into());
        }

        let chunk_target_samples = end - offset;
        let (v_left, v_right) = perform_istft(&vocals_stft, chunk_target_samples);
        let (i_left, i_right) = perform_istft(&inst_stft, chunk_target_samples);

        for i in 0..chunk_target_samples {
            final_vocals_left[offset + i] = v_left[i];
            final_vocals_right[offset + i] = v_right[i];
            final_inst_left[offset + i] = i_left[i];
            final_inst_right[offset + i] = i_right[i];
        }

        offset += CHUNK_SIZE_SAMPLES;
    }

    let file_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");

    if action != SeparationAction::RemoveVocals {
        let vocals_path = output_dir.join(format!("{}_vocals.wav", file_stem));
        save_stereo_wav(
            &vocals_path,
            &final_vocals_left,
            &final_vocals_right,
            sample_rate,
        )?;
    }

    if action != SeparationAction::GetVocals {
        let instruments_path = output_dir.join(format!("{}_instruments.wav", file_stem));
        save_stereo_wav(
            &instruments_path,
            &final_inst_left,
            &final_inst_right,
            sample_rate,
        )?;
    }

    Ok(())
}

fn create_hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size as f32 - 1.0)).cos()))
        .collect()
}

fn get_frame(data: &[f32], center: usize, window: &[f32]) -> Vec<Complex<f32>> {
    let half_window = window.len() / 2;
    let mut frame = vec![Complex::new(0.0, 0.0); window.len()];

    for i in 0..window.len() {
        let data_idx = center as isize + i as isize - half_window as isize;
        if data_idx >= 0 && data_idx < data.len() as isize {
            frame[i] = Complex::new(data[data_idx as usize] * window[i], 0.0);
        }
    }
    frame
}

fn perform_stft(left: &[f32], right: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let window = create_hann_window(N_FFT);
    let mut tensor = vec![0.0f32; STFT_CHANNELS * STFT_BINS * STFT_FRAMES];

    for t in 0..STFT_FRAMES {
        let center = t * HOP_LENGTH;
        let mut frame_l = get_frame(left, center, &window);
        let mut frame_r = get_frame(right, center, &window);
        fft.process(&mut frame_l);
        fft.process(&mut frame_r);

        for f in 0..STFT_BINS {
            let l_c = frame_l[f];
            let r_c = frame_r[f];

            let idx_l_r = f * STFT_FRAMES + t;
            let idx_l_i = STFT_BINS * STFT_FRAMES + f * STFT_FRAMES + t;
            let idx_r_r = 2 * STFT_BINS * STFT_FRAMES + f * STFT_FRAMES + t;
            let idx_r_i = 3 * STFT_BINS * STFT_FRAMES + f * STFT_FRAMES + t;

            tensor[idx_l_r] = l_c.re;
            tensor[idx_l_i] = l_c.im;
            tensor[idx_r_r] = r_c.re;
            tensor[idx_r_i] = r_c.im;
        }
    }

    tensor
}

fn perform_istft(stft_data: &[f32], target_samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(N_FFT);
    let window = create_hann_window(N_FFT);

    let mut left_out = vec![0.0f32; target_samples];
    let mut right_out = vec![0.0f32; target_samples];
    let mut window_sum = vec![0.0f32; target_samples];

    for t in 0..STFT_FRAMES {
        let center = t * HOP_LENGTH;
        let mut frame_l = vec![Complex::new(0.0, 0.0); N_FFT];
        let mut frame_r = vec![Complex::new(0.0, 0.0); N_FFT];

        for f in 0..STFT_BINS {
            let idx_l_r = f * STFT_FRAMES + t;
            let idx_l_i = STFT_BINS * STFT_FRAMES + f * STFT_FRAMES + t;
            let idx_r_r = 2 * STFT_BINS * STFT_FRAMES + f * STFT_FRAMES + t;
            let idx_r_i = 3 * STFT_BINS * STFT_FRAMES + f * STFT_FRAMES + t;

            let l_val = Complex::new(stft_data[idx_l_r], stft_data[idx_l_i]);
            let r_val = Complex::new(stft_data[idx_r_r], stft_data[idx_r_i]);
            frame_l[f] = l_val;
            frame_r[f] = r_val;

            if f > 0 && f < N_FFT / 2 {
                frame_l[N_FFT - f] = l_val.conj();
                frame_r[N_FFT - f] = r_val.conj();
            }
        }

        ifft.process(&mut frame_l);
        ifft.process(&mut frame_r);
        let half_window = window.len() / 2;
        let scale = 1.0 / N_FFT as f32;

        for i in 0..window.len() {
            let data_idx = center as isize + i as isize - half_window as isize;
            if data_idx >= 0 && data_idx < target_samples as isize {
                let idx = data_idx as usize;
                let w = window[i];
                left_out[idx] += frame_l[i].re * scale * w;
                right_out[idx] += frame_r[i].re * scale * w;
                window_sum[idx] += w * w;
            }
        }
    }

    for i in 0..target_samples {
        if window_sum[i] > 1e-7 {
            left_out[i] /= window_sum[i];
            right_out[i] /= window_sum[i];
        }
    }

    (left_out, right_out)
}

fn save_stereo_wav(
    path: &Path,
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if left.len() != right.len() {
        return Err("Stereo channel lengths differ".into());
    }

    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for (&l, &r) in left.iter().zip(right.iter()) {
        writer.write_sample(l)?;
        writer.write_sample(r)?;
    }
    writer.finalize()?;
    Ok(())
}
