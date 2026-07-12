use crate::{
    session,
    shell::{BackgroundFramePacing, IsolatedInputSeat, create_seat},
    state::State,
    utils,
};
use anyhow::{Context, Result, bail};
use calloop::{LoopHandle, RegistrationToken};
use futures_executor::ThreadPool;
use std::{
    collections::HashMap,
    os::unix::process::CommandExt,
    process,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};
use tracing::warn;

const SERVICE_NAME: &str = "com.system76.CosmicComp.BackgroundLaunch";
const OBJECT_PATH: &str = "/com/system76/CosmicComp/BackgroundLaunch";

static NEXT_LAUNCH_ID: AtomicU64 = AtomicU64::new(1);

pub struct LaunchRequest {
    pub workspace_name: String,
    pub argv: Vec<String>,
    pub cwd: String,
    pub env: HashMap<String, String>,
    pub frame_pacing: BackgroundFramePacing,
    pub isolated_input: bool,
    pub reply: mpsc::Sender<Result<LaunchReply, String>>,
}

pub struct ReconcileRequest {
    pub launch_id: String,
    pub reply: mpsc::Sender<Result<u32, String>>,
}

pub struct IsolationStatusRequest {
    pub launch_id: String,
    pub reply: mpsc::Sender<Result<IsolationStatusReply, String>>,
}

pub struct ReleaseRequest {
    pub launch_id: String,
    pub reply: mpsc::Sender<Result<(), String>>,
}

pub enum BackgroundRequest {
    Launch(LaunchRequest),
    Reconcile(ReconcileRequest),
    IsolationStatus(IsolationStatusRequest),
    Release(ReleaseRequest),
}

pub struct LaunchReply {
    pub pid: u32,
    pub launch_id: String,
    pub isolated_seat_name: Option<String>,
}

pub type IsolationStatusReply = (
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
);

struct BackgroundLaunch {
    tx: calloop::channel::Sender<BackgroundRequest>,
}

impl BackgroundLaunch {
    fn request_launch(
        &self,
        workspace_name: String,
        argv: Vec<String>,
        cwd: String,
        env: HashMap<String, String>,
        frame_pacing: BackgroundFramePacing,
        isolated_input: bool,
    ) -> zbus::fdo::Result<(u32, String)> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(BackgroundRequest::Launch(LaunchRequest {
                workspace_name,
                argv,
                cwd,
                env,
                frame_pacing,
                isolated_input,
                reply,
            }))
            .map_err(|err| zbus::fdo::Error::Failed(format!("compositor unavailable: {err}")))?;

        let reply = rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|err| zbus::fdo::Error::Failed(format!("launch timed out: {err}")))?;
        reply
            .map(|reply| (reply.pid, reply.launch_id))
            .map_err(zbus::fdo::Error::Failed)
    }
}

#[zbus::interface(name = "com.system76.CosmicComp.BackgroundLaunch1")]
impl BackgroundLaunch {
    async fn launch(
        &self,
        workspace_name: String,
        argv: Vec<String>,
        cwd: String,
        env: HashMap<String, String>,
    ) -> zbus::fdo::Result<(u32, String)> {
        self.request_launch(
            workspace_name,
            argv,
            cwd,
            env,
            BackgroundFramePacing::Standard,
            false,
        )
    }

    async fn launch_with_options(
        &self,
        workspace_name: String,
        argv: Vec<String>,
        cwd: String,
        env: HashMap<String, String>,
        frame_pacing: String,
    ) -> zbus::fdo::Result<(u32, String)> {
        let frame_pacing = frame_pacing
            .parse()
            .map_err(zbus::fdo::Error::InvalidArgs)?;
        self.request_launch(workspace_name, argv, cwd, env, frame_pacing, false)
    }

    async fn launch_isolated(
        &self,
        workspace_name: String,
        argv: Vec<String>,
        cwd: String,
        env: HashMap<String, String>,
        frame_pacing: String,
    ) -> zbus::fdo::Result<(u32, String, String)> {
        let frame_pacing = frame_pacing
            .parse()
            .map_err(zbus::fdo::Error::InvalidArgs)?;
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(BackgroundRequest::Launch(LaunchRequest {
                workspace_name,
                argv,
                cwd,
                env,
                frame_pacing,
                isolated_input: true,
                reply,
            }))
            .map_err(|err| zbus::fdo::Error::Failed(format!("compositor unavailable: {err}")))?;
        let reply = rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|err| zbus::fdo::Error::Failed(format!("launch timed out: {err}")))?
            .map_err(zbus::fdo::Error::Failed)?;
        let seat_name = reply.isolated_seat_name.ok_or_else(|| {
            zbus::fdo::Error::Failed("isolated launch omitted its seat name".to_string())
        })?;
        Ok((reply.pid, reply.launch_id, seat_name))
    }

    async fn reconcile(&self, launch_id: String) -> zbus::fdo::Result<u32> {
        if launch_id.trim().is_empty() {
            return Err(zbus::fdo::Error::InvalidArgs(
                "launch ID must not be empty".to_string(),
            ));
        }
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(BackgroundRequest::Reconcile(ReconcileRequest {
                launch_id,
                reply,
            }))
            .map_err(|err| zbus::fdo::Error::Failed(format!("compositor unavailable: {err}")))?;
        rx.recv_timeout(Duration::from_secs(5))
            .map_err(|err| zbus::fdo::Error::Failed(format!("reconcile timed out: {err}")))?
            .map_err(zbus::fdo::Error::Failed)
    }

    async fn isolation_status(&self, launch_id: String) -> zbus::fdo::Result<IsolationStatusReply> {
        if launch_id.trim().is_empty() {
            return Err(zbus::fdo::Error::InvalidArgs(
                "launch ID must not be empty".to_string(),
            ));
        }
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(BackgroundRequest::IsolationStatus(IsolationStatusRequest {
                launch_id,
                reply,
            }))
            .map_err(|err| zbus::fdo::Error::Failed(format!("compositor unavailable: {err}")))?;
        rx.recv_timeout(Duration::from_secs(5))
            .map_err(|err| zbus::fdo::Error::Failed(format!("status timed out: {err}")))?
            .map_err(zbus::fdo::Error::Failed)
    }

    async fn release(&self, launch_id: String) -> zbus::fdo::Result<()> {
        if launch_id.trim().is_empty() {
            return Err(zbus::fdo::Error::InvalidArgs(
                "launch ID must not be empty".to_string(),
            ));
        }
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(BackgroundRequest::Release(ReleaseRequest {
                launch_id,
                reply,
            }))
            .map_err(|err| zbus::fdo::Error::Failed(format!("compositor unavailable: {err}")))?;
        rx.recv_timeout(Duration::from_secs(5))
            .map_err(|err| zbus::fdo::Error::Failed(format!("release timed out: {err}")))?
            .map_err(zbus::fdo::Error::Failed)
    }
}

pub fn init(evlh: &LoopHandle<'static, State>, executor: &ThreadPool) -> Result<RegistrationToken> {
    let (tx, rx) = calloop::channel::channel();
    let token = evlh
        .insert_source(rx, |event, _, state| match event {
            calloop::channel::Event::Msg(BackgroundRequest::Launch(request)) => {
                state.handle_background_launch(request)
            }
            calloop::channel::Event::Msg(BackgroundRequest::Reconcile(request)) => {
                state.handle_background_reconcile(request)
            }
            calloop::channel::Event::Msg(BackgroundRequest::IsolationStatus(request)) => {
                state.handle_background_isolation_status(request)
            }
            calloop::channel::Event::Msg(BackgroundRequest::Release(request)) => {
                state.handle_background_release(request)
            }
            calloop::channel::Event::Closed => (),
        })
        .map_err(|err| err.error)
        .with_context(|| "Failed to add background launch channel to event_loop")?;

    executor.spawn_ok(async move {
        if let Err(err) = serve(tx).await {
            warn!(?err, "Failed to serve background launch D-Bus API");
        }
    });

    Ok(token)
}

async fn serve(tx: calloop::channel::Sender<BackgroundRequest>) -> zbus::Result<()> {
    let _connection = zbus::connection::Builder::session()?
        .name(SERVICE_NAME)?
        .serve_at(OBJECT_PATH, BackgroundLaunch { tx })?
        .build()
        .await?;
    std::future::pending::<()>().await;
    Ok(())
}

impl State {
    pub fn handle_background_launch(&mut self, request: LaunchRequest) {
        let result = self
            .launch_background_app(
                &request.workspace_name,
                &request.argv,
                &request.cwd,
                &request.env,
                request.frame_pacing,
                request.isolated_input,
            )
            .map_err(|err| err.to_string());
        let _ = request.reply.send(result);
    }

    pub fn handle_background_reconcile(&mut self, request: ReconcileRequest) {
        let result = self
            .common
            .shell
            .write()
            .reconcile_background_launch(
                &request.launch_id,
                &self.common.display_handle,
                &mut self.common.workspace_state.update(),
                &self.common.event_loop_handle,
            )
            .map(|count| u32::try_from(count).unwrap_or(u32::MAX));
        let _ = request.reply.send(result);
    }

    pub fn handle_background_isolation_status(&mut self, request: IsolationStatusRequest) {
        let result = self
            .common
            .shell
            .read()
            .background_input_isolation_status(&request.launch_id)
            .map(|status| {
                (
                    status.seat_name,
                    u32::try_from(status.device_count).unwrap_or(u32::MAX),
                    status.isolated_pointer_x,
                    status.isolated_pointer_y,
                    status.physical_pointer_x,
                    status.physical_pointer_y,
                    status.workspace_active,
                    u32::try_from(status.mapped_surface_count).unwrap_or(u32::MAX),
                    status.tiling_enabled,
                    u32::try_from(status.floating_window_count).unwrap_or(u32::MAX),
                    u32::try_from(status.tiled_window_count).unwrap_or(u32::MAX),
                    u32::try_from(status.maximized_window_count).unwrap_or(u32::MAX),
                )
            });
        let _ = request.reply.send(result);
    }

    pub fn handle_background_release(&mut self, request: ReleaseRequest) {
        let result = self
            .common
            .shell
            .write()
            .release_background_launch(&request.launch_id, &self.common.display_handle);
        let _ = request.reply.send(result);
    }

    fn launch_background_app(
        &mut self,
        workspace_name: &str,
        argv: &[String],
        cwd: &str,
        env: &HashMap<String, String>,
        frame_pacing: BackgroundFramePacing,
        isolated_input: bool,
    ) -> Result<LaunchReply> {
        if workspace_name.trim().is_empty() {
            bail!("workspace name must not be empty");
        }
        let (program, args) = argv.split_first().context("argv must not be empty")?;
        if program.is_empty() {
            bail!("program must not be empty");
        }

        let launch_id = format!(
            "background-launch-{}-{}",
            process::id(),
            NEXT_LAUNCH_ID.fetch_add(1, Ordering::Relaxed)
        );

        let (workspace_handle, output) = {
            let mut shell = self.common.shell.write();
            let workspace_handle = shell
                .ensure_background_launch_workspace(
                    workspace_name,
                    &mut self.common.workspace_state.update(),
                )
                .context("background workspace has no output")?;
            let output = shell
                .workspaces
                .space_for_handle(&workspace_handle)
                .context("background workspace disappeared")?
                .output
                .clone();
            (workspace_handle, output)
        };

        let isolated_seat = isolated_input.then(|| {
            let name = format!("cosmic-isolated-{}", launch_id);
            let seat = create_seat(
                &self.common.display_handle,
                &mut self.common.seat_state,
                &output,
                &self.common.config,
                name.clone(),
            );
            seat.user_data()
                .insert_if_missing_threadsafe(|| IsolatedInputSeat {
                    launch_id: launch_id.clone(),
                    name,
                    workspace: workspace_handle,
                });
            self.common.shell.write().seats.add_seat(seat.clone());
            seat
        });
        let isolated_seat_name = isolated_seat.as_ref().map(|seat| {
            seat.user_data()
                .get::<IsolatedInputSeat>()
                .expect("isolated seat metadata")
                .name
                .clone()
        });

        let mut command = process::Command::new(program);
        command.args(args);
        if !cwd.is_empty() {
            command.current_dir(cwd);
        }
        command.envs(session::get_env(&self.common)?);
        command.envs(env);
        command.env("COSMIC_BACKGROUND_LAUNCH_ID", &launch_id);
        command.env("XDG_ACTIVATION_TOKEN", &launch_id);
        command.env("DESKTOP_STARTUP_ID", &launch_id);
        if let Some(seat_name) = &isolated_seat_name {
            command.env("APP_WINDOW_WAYLAND_SEAT", seat_name);
        }
        unsafe {
            command.pre_exec(|| {
                utils::rlimit::restore_nofile_limit();
                Ok(())
            })
        };

        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                if let Some(seat) = &isolated_seat {
                    self.common.shell.write().seats.remove_seat(seat);
                    if let Some(global) = seat.global() {
                        self.common.display_handle.remove_global::<State>(global);
                    }
                }
                return Err(error).with_context(|| format!("failed to spawn {program:?}"));
            }
        };
        let pid = child.id();
        self.common.background_launch_children.push(child);

        self.common.shell.write().register_background_launch(
            launch_id.clone(),
            workspace_name.to_string(),
            pid,
            frame_pacing,
            isolated_seat,
            &mut self.common.workspace_state.update(),
        );

        Ok(LaunchReply {
            pid,
            launch_id,
            isolated_seat_name,
        })
    }
}
