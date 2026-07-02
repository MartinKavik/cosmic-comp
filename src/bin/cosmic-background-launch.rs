use std::{collections::HashMap, env, process};

use zbus::blocking::{Connection, Proxy};

const SERVICE_NAME: &str = "com.system76.CosmicComp.BackgroundLaunch";
const OBJECT_PATH: &str = "/com/system76/CosmicComp/BackgroundLaunch";
const INTERFACE_NAME: &str = "com.system76.CosmicComp.BackgroundLaunch1";

fn usage() -> ! {
    eprintln!("usage: cosmic-background-launch --workspace <name> -- <command> [args...]");
    process::exit(2);
}

fn main() -> zbus::Result<()> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("--workspace") {
        usage();
    }
    let Some(workspace_name) = args.next().filter(|name| !name.trim().is_empty()) else {
        usage();
    };
    if args.next().as_deref() != Some("--") {
        usage();
    }

    let argv = args.collect::<Vec<_>>();
    if argv.is_empty() {
        usage();
    }

    let cwd = env::current_dir()
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
        .unwrap_or_default();

    let conn = Connection::session()?;
    let proxy = Proxy::new(&conn, SERVICE_NAME, OBJECT_PATH, INTERFACE_NAME)?;
    let env = HashMap::<String, String>::new();
    let (pid, launch_id): (u32, String) =
        proxy.call("Launch", &(workspace_name, argv, cwd, env))?;
    println!("{pid} {launch_id}");

    Ok(())
}
