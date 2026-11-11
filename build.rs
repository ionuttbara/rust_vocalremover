fn main() {
    println!("cargo:rerun-if-changed=assets/app_icon.ico");

    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("windows") {
        let mut resource = winresource::WindowsResource::new();
        resource
            .set_icon("assets/app_icon.ico")
            .set("ProductName", "Vocal Remover AI")
            .set("FileDescription", "AI Vocal Remover")
            .set("InternalName", "rustvocalremover.exe")
            .set("OriginalFilename", "rustvocalremover.exe")
            .set("FileVersion", "0.2.2")
            .set("ProductVersion", "0.2.2");
        resource.compile().expect("failed to embed Windows application icon");
    }
}
