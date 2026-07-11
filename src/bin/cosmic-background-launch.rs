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
    argv: Vec<String>,
}

fn usage() -> ! {
    eprintln!(
        "usage: cosmic-background-launch --workspace <name> \
         [--frame-pacing standard|demand] -- <command> [args...]"
    );
    process::exit(2);
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Option<LaunchArgs> {
    let mut args = args.into_iter();
    let mut workspace_name = None;
    let mut frame_pacing = None;

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
            _ => return None,
        }
    }

    let argv = args.collect::<Vec<_>>();
    if argv.is_empty() {
        return None;
    }

    Some(LaunchArgs {
        workspace_name: workspace_name?,
        frame_pacing: frame_pacing.unwrap_or_else(|| DEFAULT_FRAME_PACING.to_string()),
        argv,
    })
}

fn main() -> zbus::Result<()> {
    let Some(LaunchArgs {
        workspace_name,
        frame_pacing,
        argv,
    }) = parse_args(env::args().skip(1))
    else {
        usage();
    };

    let cwd = env::current_dir()
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
        .unwrap_or_default();

    let conn = Connection::session()?;
    let proxy = Proxy::new(&conn, SERVICE_NAME, OBJECT_PATH, INTERFACE_NAME)?;
    let launch_env = HashMap::<String, String>::new();
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

    fn args(values: &[&str]) -> Option<LaunchArgs> {
        parse_args(values.iter().map(|value| value.to_string()))
    }

    #[test]
    fn parses_standard_pacing_by_default() {
        assert_eq!(
            args(&["--workspace", "build", "--", "command", "arg"]),
            Some(LaunchArgs {
                workspace_name: "build".to_string(),
                frame_pacing: "standard".to_string(),
                argv: vec!["command".to_string(), "arg".to_string()],
            })
        );
    }

    #[test]
    fn parses_demand_pacing_in_either_option_order() {
        let expected = Some(LaunchArgs {
            workspace_name: "render".to_string(),
            frame_pacing: "demand".to_string(),
            argv: vec!["command".to_string()],
        });
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
}
