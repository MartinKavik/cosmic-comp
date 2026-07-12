use crate::{session, shell::BackgroundFramePacing, state::State, utils};
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
    pub reply: mpsc::Sender<Result<LaunchReply, String>>,
}

pub struct ReconcileRequest {
    pub launch_id: String,
    pub reply: mpsc::Sender<Result<u32, String>>,
}

pub enum BackgroundRequest {
    Launch(LaunchRequest),
    Reconcile(ReconcileRequest),
}

pub struct LaunchReply {
    pub pid: u32,
    pub launch_id: String,
}

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
    ) -> zbus::fdo::Result<(u32, String)> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(BackgroundRequest::Launch(LaunchRequest {
                workspace_name,
                argv,
                cwd,
                env,
                frame_pacing,
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
        self.request_launch(workspace_name, argv, cwd, env, frame_pacing)
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

    fn launch_background_app(
        &mut self,
        workspace_name: &str,
        argv: &[String],
        cwd: &str,
        env: &HashMap<String, String>,
        frame_pacing: BackgroundFramePacing,
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

        {
            let mut shell = self.common.shell.write();
            let _ = shell.ensure_background_launch_workspace(
                workspace_name,
                &mut self.common.workspace_state.update(),
            );
        }

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
        unsafe {
            command.pre_exec(|| {
                utils::rlimit::restore_nofile_limit();
                Ok(())
            })
        };

        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn {program:?}"))?;
        let pid = child.id();
        self.common.background_launch_children.push(child);

        self.common.shell.write().register_background_launch(
            launch_id.clone(),
            workspace_name.to_string(),
            pid,
            frame_pacing,
            &mut self.common.workspace_state.update(),
        );

        Ok(LaunchReply { pid, launch_id })
    }
}
