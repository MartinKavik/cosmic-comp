use std::{collections::HashMap, env, process};

use zbus::blocking::{Connection, Proxy};

const SERVICE_NAME: &str = "com.system76.CosmicComp.BackgroundLaunch";
const OBJECT_PATH: &str = "/com/system76/CosmicComp/BackgroundLaunch";
const INTERFACE_NAME: &str = "com.system76.CosmicComp.BackgroundLaunch1";
const DEFAULT_FRAME_PACING: &str = "standard";

#[derive(Clone, Debug, PartialEq, Eq)]
struct LaunchArgs {
    workspace_name: String,
    frame_pacing: String,
    isolated_input: bool,
    argv: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CommandArgs {
    Launch(LaunchArgs),
    Reconcile { launch_id: String },
    IsolationStatus { launch_id: String },
    Release { launch_id: String },
}

fn usage() -> ! {
    eprintln!(
        "usage: cosmic-background-launch --workspace <name> \
         [--frame-pacing standard|demand] [--isolated-input] -- <command> [args...]\n       \
         cosmic-background-launch --reconcile <launch-id>\n       \
         cosmic-background-launch --isolation-status <launch-id>\n       \
         cosmic-background-launch --release <launch-id>"
    );
    process::exit(2);
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Option<CommandArgs> {
    let mut args = args.into_iter().peekable();
    if args.peek().is_some_and(|arg| arg == "--reconcile") {
        args.next();
        let launch_id = args.next().filter(|value| !value.trim().is_empty())?;
        if args.next().is_some() {
            return None;
        }
        return Some(CommandArgs::Reconcile { launch_id });
    }
    if args.peek().is_some_and(|arg| arg == "--isolation-status") {
        args.next();
        let launch_id = args.next().filter(|value| !value.trim().is_empty())?;
        if args.next().is_some() {
            return None;
        }
        return Some(CommandArgs::IsolationStatus { launch_id });
    }
    if args.peek().is_some_and(|arg| arg == "--release") {
        args.next();
        let launch_id = args.next().filter(|value| !value.trim().is_empty())?;
        if args.next().is_some() {
            return None;
        }
        return Some(CommandArgs::Release { launch_id });
    }
    let mut workspace_name = None;
    let mut frame_pacing = None;
    let mut isolated_input = false;

    loop {
        match args.next()?.as_str() {
            "--" => break,
            "--workspace" if workspace_name.is_none() => {
                workspace_name = args.next().filter(|name| !name.trim().is_empty());
                workspace_name.as_ref()?;
            }
            "--frame-pacing" if frame_pacing.is_none() => {
                let value = args.next()?;
                if !matches!(value.as_str(), "standard" | "demand") {
                    return None;
                }
                frame_pacing = Some(value);
            }
            "--isolated-input" if !isolated_input => isolated_input = true,
            _ => return None,
        }
    }

    let argv = args.collect::<Vec<_>>();
    if argv.is_empty() {
        return None;
    }

    Some(CommandArgs::Launch(LaunchArgs {
        workspace_name: workspace_name?,
        frame_pacing: frame_pacing.unwrap_or_else(|| DEFAULT_FRAME_PACING.to_string()),
        isolated_input,
        argv,
    }))
}

fn main() -> zbus::Result<()> {
    let Some(command) = parse_args(env::args().skip(1)) else {
        usage();
    };

    let conn = Connection::session()?;
    let proxy = Proxy::new(&conn, SERVICE_NAME, OBJECT_PATH, INTERFACE_NAME)?;
    let LaunchArgs {
        workspace_name,
        frame_pacing,
        isolated_input,
        argv,
    } = match command {
        CommandArgs::Reconcile { launch_id } => {
            let count: u32 = proxy.call("Reconcile", &(launch_id,))?;
            println!("{count}");
            return Ok(());
        }
        CommandArgs::IsolationStatus { launch_id } => {
            let (
                seat_name,
                device_count,
                isolated_pointer_x,
                isolated_pointer_y,
                physical_pointer_x,
                physical_pointer_y,
                workspace_active,
                mapped_surface_count,
                tiling_enabled,
                floating_window_count,
                tiled_window_count,
                maximized_window_count,
            ): (
                String,
                u32,
                f64,
                f64,
                f64,
                f64,
                bool,
                u32,
                bool,
                u32,
                u32,
                u32,
            ) = proxy.call("IsolationStatus", &(launch_id,))?;
            println!(
                "seat={seat_name} devices={device_count} isolated_x={isolated_pointer_x:.3} \
                 isolated_y={isolated_pointer_y:.3} physical_x={physical_pointer_x:.3} \
                 physical_y={physical_pointer_y:.3} workspace_active={workspace_active} \
                 mapped_surfaces={mapped_surface_count} tiling_enabled={tiling_enabled} \
                 floating_windows={floating_window_count} tiled_windows={tiled_window_count} \
                 maximized_windows={maximized_window_count}"
            );
            return Ok(());
        }
        CommandArgs::Release { launch_id } => {
            let (): () = proxy.call("Release", &(launch_id.clone(),))?;
            println!("released {launch_id}");
            return Ok(());
        }
        CommandArgs::Launch(args) => args,
    };

    let cwd = env::current_dir()
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
        .unwrap_or_default();

    let launch_env = HashMap::<String, String>::new();
    if isolated_input {
        let (pid, launch_id, seat_name): (u32, String, String) = proxy.call(
            "LaunchIsolated",
            &(workspace_name, argv, cwd, launch_env, frame_pacing),
        )?;
        println!("{pid} {launch_id} {seat_name}");
        return Ok(());
    }
    let (pid, launch_id): (u32, String) = if frame_pacing == DEFAULT_FRAME_PACING {
        proxy.call("Launch", &(workspace_name, argv, cwd, launch_env))?
    } else {
        proxy.call(
            "LaunchWithOptions",
            &(workspace_name, argv, cwd, launch_env, frame_pacing),
        )?
    };
    println!("{pid} {launch_id}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Option<CommandArgs> {
        parse_args(values.iter().map(|value| value.to_string()))
    }

    #[test]
    fn parses_standard_pacing_by_default() {
        assert_eq!(
            args(&["--workspace", "build", "--", "command", "arg"]),
            Some(CommandArgs::Launch(LaunchArgs {
                workspace_name: "build".to_string(),
                frame_pacing: "standard".to_string(),
                isolated_input: false,
                argv: vec!["command".to_string(), "arg".to_string()],
            }))
        );
    }

    #[test]
    fn parses_demand_pacing_in_either_option_order() {
        let expected = Some(CommandArgs::Launch(LaunchArgs {
            workspace_name: "render".to_string(),
            frame_pacing: "demand".to_string(),
            isolated_input: false,
            argv: vec!["command".to_string()],
        }));
        assert_eq!(
            args(&[
                "--workspace",
                "render",
                "--frame-pacing",
                "demand",
                "--",
                "command"
            ]),
            expected.clone()
        );
        assert_eq!(
            args(&[
                "--frame-pacing",
                "demand",
                "--workspace",
                "render",
                "--",
                "command"
            ]),
            expected
        );
    }

    #[test]
    fn rejects_unknown_pacing_and_missing_command() {
        assert_eq!(
            args(&[
                "--workspace",
                "render",
                "--frame-pacing",
                "realtime",
                "--",
                "command"
            ]),
            None
        );
        assert_eq!(args(&["--workspace", "render", "--"]), None);
    }

    #[test]
    fn parses_only_one_nonempty_reconcile_id() {
        assert_eq!(
            args(&["--reconcile", "background-launch-7"]),
            Some(CommandArgs::Reconcile {
                launch_id: "background-launch-7".to_string()
            })
        );
        assert_eq!(args(&["--reconcile", ""]), None);
        assert_eq!(args(&["--reconcile", "id", "extra"]), None);
    }

    #[test]
    fn parses_isolated_input_once() {
        assert_eq!(
            args(&["--workspace", "render", "--isolated-input", "--", "command"]),
            Some(CommandArgs::Launch(LaunchArgs {
                workspace_name: "render".to_string(),
                frame_pacing: "standard".to_string(),
                isolated_input: true,
                argv: vec!["command".to_string()],
            }))
        );
        assert_eq!(
            args(&[
                "--workspace",
                "render",
                "--isolated-input",
                "--isolated-input",
                "--",
                "command"
            ]),
            None
        );
    }

    #[test]
    fn parses_isolation_status_and_release() {
        assert_eq!(
            args(&["--isolation-status", "background-launch-7"]),
            Some(CommandArgs::IsolationStatus {
                launch_id: "background-launch-7".to_string()
            })
        );
        assert_eq!(
            args(&["--release", "background-launch-7"]),
            Some(CommandArgs::Release {
                launch_id: "background-launch-7".to_string()
            })
        );
        assert_eq!(args(&["--isolation-status", ""]), None);
        assert_eq!(args(&["--release", "id", "extra"]), None);
    }
}
