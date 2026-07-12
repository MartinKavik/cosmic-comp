#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::len_without_is_empty
)]
// SPDX-License-Identifier: GPL-3.0-only

use calloop::timer::{TimeoutAction, Timer};
use smithay::{
    reexports::{
        calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic},
        wayland_server::{Display, DisplayHandle},
    },
    wayland::socket::ListeningSocketSource,
};

use anyhow::{Context, Result};
use state::{LastRefresh, State};
use std::{
    env,
    ffi::OsString,
    os::unix::process::CommandExt,
    process,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use wayland::protocols::overlap_notify::OverlapNotifyState;

use crate::wayland::handlers::compositor::client_compositor_state;

use clap_lex::RawArgs;

use std::error::Error;

pub mod backend;
pub mod config;
pub mod dbus;
#[cfg(feature = "debug")]
pub mod debug;
pub mod hooks;
pub mod input;
mod logger;
pub mod session;
pub mod shell;
pub mod state;
#[cfg(feature = "systemd")]
pub mod systemd;
pub mod theme;
pub mod utils;
pub mod wayland;
pub mod xwayland;

#[cfg(feature = "profile-with-tracy")]
#[global_allocator]
static GLOBAL: profiling::tracy_client::ProfiledAllocator<std::alloc::System> =
    profiling::tracy_client::ProfiledAllocator::new(std::alloc::System, 10);

static MAIN_LOOP_STATS: LazyLock<Mutex<MainLoopStats>> =
    LazyLock::new(|| Mutex::new(MainLoopStats::default()));

const COMPOSITOR_TARGET_NICE: i32 = -10;

#[derive(Default)]
struct MainLoopCounters {
    callbacks: u64,
    callback_us_total: u64,
    callback_us_max: u64,
    update_animations_us_total: u64,
    update_animations_us_max: u64,
    blocker_clear_us_total: u64,
    blocker_clear_us_max: u64,
    refresh_us_total: u64,
    refresh_us_max: u64,
    animation_schedule_us_total: u64,
    animation_schedule_us_max: u64,
    flush_clients_us_total: u64,
    flush_clients_us_max: u64,
    dispatch_clients_calls: u64,
    dispatch_clients_us_total: u64,
    dispatch_clients_us_max: u64,
    dispatch_clients_events_total: u64,
    animation_outputs_scheduled: u64,
    animation_active_callbacks: u64,
}

struct MainLoopStats {
    last_log: Instant,
    counters: MainLoopCounters,
}

impl Default for MainLoopStats {
    fn default() -> Self {
        Self {
            last_log: Instant::now(),
            counters: MainLoopCounters::default(),
        }
    }
}

struct MainLoopSample {
    callback: Duration,
    update_animations: Duration,
    blocker_clear: Duration,
    refresh: Duration,
    animation_schedule: Duration,
    flush_clients: Duration,
    animation_outputs_scheduled: u64,
    animations_active: bool,
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn note_main_loop_sample(sample: MainLoopSample) {
    let mut stats = MAIN_LOOP_STATS.lock().unwrap();
    let counters = &mut stats.counters;

    counters.callbacks = counters.callbacks.saturating_add(1);

    let callback_us = duration_us(sample.callback);
    counters.callback_us_total = counters.callback_us_total.saturating_add(callback_us);
    counters.callback_us_max = counters.callback_us_max.max(callback_us);

    let update_us = duration_us(sample.update_animations);
    counters.update_animations_us_total = counters
        .update_animations_us_total
        .saturating_add(update_us);
    counters.update_animations_us_max = counters.update_animations_us_max.max(update_us);

    let blocker_us = duration_us(sample.blocker_clear);
    counters.blocker_clear_us_total = counters.blocker_clear_us_total.saturating_add(blocker_us);
    counters.blocker_clear_us_max = counters.blocker_clear_us_max.max(blocker_us);

    let refresh_us = duration_us(sample.refresh);
    counters.refresh_us_total = counters.refresh_us_total.saturating_add(refresh_us);
    counters.refresh_us_max = counters.refresh_us_max.max(refresh_us);

    let schedule_us = duration_us(sample.animation_schedule);
    counters.animation_schedule_us_total = counters
        .animation_schedule_us_total
        .saturating_add(schedule_us);
    counters.animation_schedule_us_max = counters.animation_schedule_us_max.max(schedule_us);

    let flush_us = duration_us(sample.flush_clients);
    counters.flush_clients_us_total = counters.flush_clients_us_total.saturating_add(flush_us);
    counters.flush_clients_us_max = counters.flush_clients_us_max.max(flush_us);

    counters.animation_outputs_scheduled = counters
        .animation_outputs_scheduled
        .saturating_add(sample.animation_outputs_scheduled);
    if sample.animations_active {
        counters.animation_active_callbacks = counters.animation_active_callbacks.saturating_add(1);
    }

    maybe_log_main_loop_stats(&mut stats);
}

fn note_wayland_dispatch_sample(elapsed: Duration, dispatched: u64) {
    let mut stats = MAIN_LOOP_STATS.lock().unwrap();
    let counters = &mut stats.counters;
    let elapsed_us = duration_us(elapsed);

    counters.dispatch_clients_calls = counters.dispatch_clients_calls.saturating_add(1);
    counters.dispatch_clients_us_total = counters
        .dispatch_clients_us_total
        .saturating_add(elapsed_us);
    counters.dispatch_clients_us_max = counters.dispatch_clients_us_max.max(elapsed_us);
    counters.dispatch_clients_events_total = counters
        .dispatch_clients_events_total
        .saturating_add(dispatched);

    maybe_log_main_loop_stats(&mut stats);
}

fn maybe_log_main_loop_stats(stats: &mut MainLoopStats) {
    if stats.last_log.elapsed() < Duration::from_secs(60) {
        return;
    }

    let counters = std::mem::take(&mut stats.counters);
    stats.last_log = Instant::now();

    let avg_callback_us = if counters.callbacks == 0 {
        0
    } else {
        counters.callback_us_total / counters.callbacks
    };
    let avg_dispatch_clients_us = if counters.dispatch_clients_calls == 0 {
        0
    } else {
        counters.dispatch_clients_us_total / counters.dispatch_clients_calls
    };

    warn!(
        callbacks = counters.callbacks,
        callback_us_total = counters.callback_us_total,
        callback_us_avg = avg_callback_us,
        callback_us_max = counters.callback_us_max,
        update_animations_us_total = counters.update_animations_us_total,
        update_animations_us_max = counters.update_animations_us_max,
        blocker_clear_us_total = counters.blocker_clear_us_total,
        blocker_clear_us_max = counters.blocker_clear_us_max,
        refresh_us_total = counters.refresh_us_total,
        refresh_us_max = counters.refresh_us_max,
        animation_schedule_us_total = counters.animation_schedule_us_total,
        animation_schedule_us_max = counters.animation_schedule_us_max,
        flush_clients_us_total = counters.flush_clients_us_total,
        flush_clients_us_max = counters.flush_clients_us_max,
        dispatch_clients_calls = counters.dispatch_clients_calls,
        dispatch_clients_events_total = counters.dispatch_clients_events_total,
        dispatch_clients_us_total = counters.dispatch_clients_us_total,
        dispatch_clients_us_avg = avg_dispatch_clients_us,
        dispatch_clients_us_max = counters.dispatch_clients_us_max,
        animation_outputs_scheduled = counters.animation_outputs_scheduled,
        animation_active_callbacks = counters.animation_active_callbacks,
        "[perf] main loop stats"
    );
}

fn raise_compositor_priority() {
    match rustix::process::getpriority_process(None) {
        Ok(current) if current <= COMPOSITOR_TARGET_NICE => {
            info!(nice = current, "Compositor CPU priority already elevated");
        }
        Ok(current) => match rustix::process::setpriority_process(None, COMPOSITOR_TARGET_NICE) {
            Ok(()) => info!(
                old_nice = current,
                new_nice = COMPOSITOR_TARGET_NICE,
                "Elevated compositor CPU priority"
            ),
            Err(err) => warn!(
                old_nice = current,
                target_nice = COMPOSITOR_TARGET_NICE,
                ?err,
                "Failed to elevate compositor CPU priority"
            ),
        },
        Err(err) => warn!(?err, "Failed to read compositor CPU priority"),
    }
}

// called by the Xwayland source, either after starting or failing
impl State {
    fn notify_ready(&mut self) {
        // TODO: Don't notify again, but potentially import updated env-variables
        // into systemd and the session?
        self.ready.call_once(|| {
            // potentially tell systemd we are setup now
            if let state::BackendData::Kms(_) = &self.backend {
                #[cfg(feature = "systemd")]
                systemd::ready(&self.common);
                if let Err(err) = dbus::ready(&self.common) {
                    error!(?err, "Failed to update the D-Bus activation environment");
                }
            }

            // potentially tell the session we are setup now
            if let Err(err) =
                session::setup_socket(self.common.event_loop_handle.clone(), &self.common)
            {
                warn!(?err, "Failed to setup cosmic-session communication");
            }

            let mut args = env::args().skip(1);
            self.common.kiosk_child = if let Some(exec) = args.next() {
                // Run command in kiosk mode
                let mut command = process::Command::new(&exec);
                command.args(args);
                command.envs(
                    session::get_env(&self.common).expect("WAYLAND_DISPLAY should be valid UTF-8"),
                );
                unsafe {
                    command.pre_exec(|| {
                        utils::rlimit::restore_nofile_limit();
                        Ok(())
                    })
                };

                info!("Running {:?}", exec);
                command
                    .spawn()
                    .map_err(|err| {
                        // TODO: replace with `inspect_err` once stable
                        error!(?err, "Error running kiosk child.");
                        err
                    })
                    .ok()
            } else {
                None
            };
        });
    }
}

pub fn run(hooks: crate::hooks::Hooks) -> Result<(), Box<dyn Error>> {
    let raw_args = RawArgs::from_args();
    let mut cursor = raw_args.cursor();
    let git_hash = option_env!("GIT_HASH").unwrap_or("unknown");

    let mut with_xwayland = true;
    // Parse the arguments
    while let Some(arg) = raw_args.next_os(&mut cursor) {
        match arg.to_str() {
            Some("--help") | Some("-h") => {
                print_help(env!("CARGO_PKG_VERSION"), git_hash);
                return Ok(());
            }
            Some("--no-xwayland") => {
                tracing::info!("Running without Xwayland");
                with_xwayland = false;
            }
            Some("--version") | Some("-V") => {
                println!(
                    "cosmic-comp {} (git commit {})",
                    env!("CARGO_PKG_VERSION"),
                    git_hash
                );
                return Ok(());
            }
            _ => {}
        }
    }

    // setup logger
    logger::init_logger()?;
    info!("Cosmic starting up!");
    raise_compositor_priority();

    profiling::register_thread!("Main Thread");
    #[cfg(feature = "profile-with-tracy")]
    tracy_client::Client::start();

    utils::rlimit::increase_nofile_limit();

    // init hook globals
    hooks::HOOKS.set(hooks)
        .expect("Hooks global has already been initialized. Running multiple instances of COSMIC in one process is not supported.");

    // init event loop
    let mut event_loop = EventLoop::try_new().with_context(|| "Failed to initialize event loop")?;
    // init wayland
    let (display, socket) = init_wayland_display(&mut event_loop)?;
    // init state
    let mut state = state::State::new(
        &display,
        socket,
        event_loop.handle(),
        event_loop.get_signal(),
        with_xwayland,
    );
    // init backend
    backend::init_backend_auto(&display, &mut event_loop, &mut state)?;

    if let Err(err) = theme::watch_theme(event_loop.handle()) {
        warn!(?err, "Failed to watch theme");
    }

    // run the event loop
    event_loop.run(None, &mut state, |state| {
        let callback_start = Instant::now();

        // shall we shut down?
        if state.common.should_stop {
            info!("Shutting down");
            state.common.event_loop_signal.stop();
            state.common.event_loop_signal.wakeup();
            return;
        }

        // trigger routines
        let update_animations_start = Instant::now();
        let clients = state.common.shell.write().update_animations();
        let update_animations_elapsed = update_animations_start.elapsed();

        let blocker_clear_start = Instant::now();
        {
            let dh = state.common.display_handle.clone();
            for client in clients.values() {
                client_compositor_state(client).blocker_cleared(state, &dh);
            }
        }
        let blocker_clear_elapsed = blocker_clear_start.elapsed();

        let refresh_start = Instant::now();
        refresh(state);
        let refresh_elapsed = refresh_start.elapsed();

        let animation_schedule_start = Instant::now();
        let mut animation_outputs_scheduled = 0;
        let mut animations_active = false;
        {
            let shell = state.common.shell.read();
            if shell.animations_going() {
                animations_active = true;
                let outputs = shell.outputs().cloned().collect::<Vec<_>>();
                std::mem::drop(shell);
                animation_outputs_scheduled = outputs.len() as u64;
                for output in outputs.into_iter() {
                    state.backend.schedule_render(&output);
                }
            } else {
                std::mem::drop(shell);
            }
        }
        let animation_schedule_elapsed = animation_schedule_start.elapsed();

        // send out events
        let flush_start = Instant::now();
        let _ = state.common.display_handle.flush_clients();
        let flush_elapsed = flush_start.elapsed();
        let callback_elapsed = callback_start.elapsed();
        state
            .common
            .shell
            .read()
            .note_main_loop_elapsed(callback_elapsed);
        note_main_loop_sample(MainLoopSample {
            callback: callback_elapsed,
            update_animations: update_animations_elapsed,
            blocker_clear: blocker_clear_elapsed,
            refresh: refresh_elapsed,
            animation_schedule: animation_schedule_elapsed,
            flush_clients: flush_elapsed,
            animation_outputs_scheduled,
            animations_active,
        });

        // check if kiosk child is running
        if let Some(child) = state.common.kiosk_child.as_mut() {
            match child.try_wait() {
                // Kiosk child exited with status
                Ok(Some(exit_status)) => {
                    info!("Command exited with status {:?}", exit_status);
                    match exit_status.code() {
                        // Exiting with the same status as the kiosk child
                        Some(code) => process::exit(code),
                        // The kiosk child exited with signal, exiting with error
                        None => process::exit(1),
                    }
                }
                // Command still running
                Ok(None) => {}
                // Kiosk child disappeared, exiting with error
                Err(err) => {
                    warn!(?err, "Failed to wait for command");
                    process::exit(1);
                }
            }
        }

        let mut exited_background_pids = Vec::new();
        state
            .common
            .background_launch_children
            .retain_mut(|child| match child.try_wait() {
                Ok(Some(exit_status)) => {
                    exited_background_pids.push(child.id());
                    info!(
                        pid = child.id(),
                        "Background-launched command exited with status {:?}", exit_status
                    );
                    false
                }
                Ok(None) => true,
                Err(err) => {
                    exited_background_pids.push(child.id());
                    warn!(
                        pid = child.id(),
                        ?err,
                        "Failed to wait for background-launched command"
                    );
                    false
                }
            });
        if !exited_background_pids.is_empty() {
            let mut shell = state.common.shell.write();
            for pid in exited_background_pids {
                if let Err(error) =
                    shell.release_background_launch_for_root_pid(pid, &state.common.display_handle)
                {
                    warn!(pid, ?error, "Failed to release background launch resources");
                }
            }
        }
    })?;

    // kill kiosk child if loop exited
    if let Some(mut child) = state.common.kiosk_child.take() {
        let _ = child.kill();
    }

    // drop eventloop & state before logger
    std::mem::drop(event_loop);
    std::mem::drop(state);

    Ok(())
}

fn print_help(version: &str, git_rev: &str) {
    println!(
        r#"cosmic-comp {version} (git commit {git_rev})
System76 <info@system76.com>

Designed for the COSMIC™ desktop environment, cosmic-comp is a Wayland Compositor.

Project home page: https://github.com/pop-os/cosmic-comp

Options:
  -h, --help          Show this message
  --no-xwayland       Run without Xwayland
  -v, --version       Show the version of cosmic-comp"#
    );
}

fn init_wayland_display(
    event_loop: &mut EventLoop<state::State>,
) -> Result<(DisplayHandle, OsString)> {
    let display = Display::new().unwrap();
    let handle = display.handle();

    let source = ListeningSocketSource::new_auto().unwrap();
    let socket_name = source.socket_name().to_os_string();
    info!("Listening on {:?}", socket_name);

    event_loop
        .handle()
        .insert_source(source, |client_stream, _, state| {
            let client_state = state.new_client_state();
            if let Err(err) = state
                .common
                .display_handle
                .insert_client(client_stream, Arc::new(client_state))
            {
                warn!(?err, "Error adding wayland client")
            };
        })
        .with_context(|| "Failed to init the wayland socket source.")?;
    event_loop
        .handle()
        .insert_source(
            Generic::new(display, Interest::READ, Mode::Level),
            move |_, display, state| {
                // SAFETY: We don't drop the display
                let dispatch_start = Instant::now();
                match unsafe { display.get_mut().dispatch_clients(state) } {
                    Ok(dispatched) => {
                        note_wayland_dispatch_sample(dispatch_start.elapsed(), dispatched as u64);
                        Ok(PostAction::Continue)
                    }
                    Err(err) => {
                        note_wayland_dispatch_sample(dispatch_start.elapsed(), 0);
                        error!(?err, "I/O error on the Wayland display");
                        state.common.should_stop = true;
                        Err(err)
                    }
                }
            },
        )
        .with_context(|| "Failed to init the wayland event source.")?;

    Ok((handle, socket_name))
}

fn refresh(state: &mut State) {
    if matches!(state.last_refresh, LastRefresh::Scheduled(_)) {
        return;
    }

    let now = Instant::now();
    let interval = Duration::from_millis(150);
    if let LastRefresh::At(instant) = state.last_refresh
        && let Some(remaining) = interval.checked_sub(now.duration_since(instant))
    {
        if let Ok(token) = state.common.event_loop_handle.insert_source(
            Timer::from_duration(remaining),
            |_, _, state| {
                state.last_refresh = LastRefresh::None;
                TimeoutAction::Drop
            },
        ) {
            state.last_refresh = LastRefresh::Scheduled(token);
            return;
        } else {
            warn!("Failed to schedule refresh");
        }
    }

    state.common.refresh();
    state::Common::refresh_focus(state);
    OverlapNotifyState::refresh(state);
    state.common.update_x11_stacking_order();
    state.last_refresh = LastRefresh::At(Instant::now());
}
