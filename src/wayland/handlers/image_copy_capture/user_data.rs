// SPDX-License-Identifier: GPL-3.0-only

use std::{
    cell::RefCell,
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use smithay::{
    backend::renderer::{
        ContextId,
        damage::OutputDamageTracker,
        gles::{GlesRenderbuffer, GlesTexture},
    },
    output::Output,
    wayland::image_copy_capture::{
        CaptureFailureReason, CursorSession, CursorSessionRef, Frame, FrameRef, Session, SessionRef,
    },
};
use tracing::warn;

use smithay::utils::user_data::UserDataMap;

use crate::shell::{CosmicSurface, Workspace};

type ImageCopySessionsData = RefCell<ImageCopySessions>;
type PendingImageCopyBuffers = Mutex<Vec<(SessionRef, Frame)>>;

pub type SessionData = Mutex<SessionUserData>;

pub struct SessionUserData {
    pub dt: OutputDamageTracker,
    pub offscreen: Option<(ContextId<GlesTexture>, GlesRenderbuffer)>,
}

impl SessionUserData {
    pub fn new(tracker: OutputDamageTracker) -> SessionUserData {
        SessionUserData {
            dt: tracker,
            offscreen: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct ImageCopySessions {
    sessions: Vec<Session>,
    cursor_sessions: Vec<CursorSession>,
    // Long-lived screencopy sessions may stay open while idle. Track recent
    // frame requests so only active captures affect the fast callback path.
    last_frame_request_ms: AtomicU64,
}

const CAPTURE_ACTIVE_TIMEOUT_MS: u64 = 1_000;
const MAX_PENDING_FRAMES_PER_OUTPUT: usize = 8;
const MAX_PENDING_FRAMES_PER_SESSION: usize = 1;
static MONOTONIC_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);
static CAPTURE_QUEUE_STATS: LazyLock<Mutex<CaptureQueueStats>> =
    LazyLock::new(|| Mutex::new(CaptureQueueStats::default()));

#[derive(Debug)]
struct CaptureQueueStats {
    last_log: Instant,
    queued: u64,
    taken: u64,
    dropped_session_cap: u64,
    dropped_output_cap: u64,
    max_pending_after_push: usize,
}

impl Default for CaptureQueueStats {
    fn default() -> Self {
        Self {
            last_log: Instant::now(),
            queued: 0,
            taken: 0,
            dropped_session_cap: 0,
            dropped_output_cap: 0,
            max_pending_after_push: 0,
        }
    }
}

fn current_time_ms() -> u64 {
    MONOTONIC_EPOCH.elapsed().as_millis() as u64
}

fn note_capture_queue(
    queued: u64,
    taken: u64,
    dropped_session_cap: u64,
    dropped_output_cap: u64,
    pending_after_push: usize,
) {
    let mut stats = CAPTURE_QUEUE_STATS.lock().unwrap();
    stats.queued = stats.queued.saturating_add(queued);
    stats.taken = stats.taken.saturating_add(taken);
    stats.dropped_session_cap = stats
        .dropped_session_cap
        .saturating_add(dropped_session_cap);
    stats.dropped_output_cap = stats.dropped_output_cap.saturating_add(dropped_output_cap);
    stats.max_pending_after_push = stats.max_pending_after_push.max(pending_after_push);

    if stats.last_log.elapsed() < std::time::Duration::from_secs(60) {
        return;
    }

    let queued = stats.queued;
    let taken = stats.taken;
    let dropped_session_cap = stats.dropped_session_cap;
    let dropped_output_cap = stats.dropped_output_cap;
    let max_pending_after_push = stats.max_pending_after_push;
    stats.queued = 0;
    stats.taken = 0;
    stats.dropped_session_cap = 0;
    stats.dropped_output_cap = 0;
    stats.max_pending_after_push = 0;
    stats.last_log = Instant::now();
    std::mem::drop(stats);

    warn!(
        queued,
        taken,
        dropped_session_cap,
        dropped_output_cap,
        max_pending_after_push,
        "[perf] capture queue stats"
    );
}

impl ImageCopySessions {
    pub fn mark_capture_active(&self) {
        self.last_frame_request_ms
            .store(current_time_ms(), Ordering::Relaxed);
    }

    pub fn is_capture_active(&self) -> bool {
        let last = self.last_frame_request_ms.load(Ordering::Relaxed);
        last != 0 && current_time_ms().saturating_sub(last) < CAPTURE_ACTIVE_TIMEOUT_MS
    }
}

/// Drop all capture sessions stored in the given `UserDataMap`.
///
/// When a toplevel is destroyed, its owned `Session` objects must be dropped
/// so that `Session::drop()` fires — this fails active frames and sends
/// `stopped` to the client, releasing GPU buffers.
///
/// Smithay doesn't automatically stop sessions when their capture source's
/// underlying toplevel dies. A cleaner long-term fix would be in smithay's
/// `ImageCopyCaptureState::cleanup()` — e.g. stopping sessions whose
/// `source().alive()` is false — so all compositors benefit without manual
/// session management. For now, we handle it on the cosmic-comp side.
pub fn stop_all_capture_sessions(user_data: &UserDataMap) {
    if let Some(data) = user_data.get::<ImageCopySessionsData>() {
        let mut data = data.borrow_mut();
        data.sessions.clear();
        data.cursor_sessions.clear();
    }
}

pub trait SessionHolder {
    fn add_session(&mut self, session: Session);
    fn remove_session(&mut self, session: &SessionRef);
    fn sessions(&self) -> Vec<SessionRef>;

    fn add_cursor_session(&mut self, session: CursorSession);
    fn remove_cursor_session(&mut self, session: &CursorSessionRef);
    fn cursor_sessions(&self) -> Vec<CursorSessionRef>;

    fn mark_capture_active(&self);
    fn is_capture_active(&self) -> bool;
}

pub trait FrameHolder {
    fn add_frame(&mut self, session: SessionRef, frame: Frame) -> bool;
    fn remove_frame(&mut self, frame: &FrameRef);
    fn take_pending_frames(&self) -> Vec<(SessionRef, Frame)>;
}

impl SessionHolder for Output {
    fn add_session(&mut self, session: Session) {
        self.user_data()
            .insert_if_missing(ImageCopySessionsData::default);
        self.user_data()
            .get::<ImageCopySessionsData>()
            .unwrap()
            .borrow_mut()
            .sessions
            .push(session);
    }

    fn remove_session(&mut self, session: &SessionRef) {
        if let Some(data) = self.user_data().get::<ImageCopySessionsData>() {
            data.borrow_mut().sessions.retain(|s| s != session);
        }
    }

    fn sessions(&self) -> Vec<SessionRef> {
        self.user_data()
            .get::<ImageCopySessionsData>()
            .map_or(Vec::new(), |sessions| {
                sessions
                    .borrow()
                    .sessions
                    .iter()
                    .map(|s| (*s).clone())
                    .collect()
            })
    }

    fn add_cursor_session(&mut self, session: CursorSession) {
        self.user_data()
            .insert_if_missing(ImageCopySessionsData::default);
        self.user_data()
            .get::<ImageCopySessionsData>()
            .unwrap()
            .borrow_mut()
            .cursor_sessions
            .push(session);
    }

    fn remove_cursor_session(&mut self, session: &CursorSessionRef) {
        if let Some(data) = self.user_data().get::<ImageCopySessionsData>() {
            data.borrow_mut().cursor_sessions.retain(|s| s != session);
        }
    }

    fn cursor_sessions(&self) -> Vec<CursorSessionRef> {
        self.user_data()
            .get::<ImageCopySessionsData>()
            .map_or(Vec::new(), |sessions| {
                sessions
                    .borrow()
                    .cursor_sessions
                    .iter()
                    .map(|s| (*s).clone())
                    .collect()
            })
    }

    fn mark_capture_active(&self) {
        if let Some(data) = self.user_data().get::<ImageCopySessionsData>() {
            data.borrow().mark_capture_active();
        }
    }

    fn is_capture_active(&self) -> bool {
        self.user_data()
            .get::<ImageCopySessionsData>()
            .map_or(false, |data| data.borrow().is_capture_active())
    }
}

impl FrameHolder for Output {
    fn add_frame(&mut self, session: SessionRef, frame: Frame) -> bool {
        self.user_data()
            .insert_if_missing_threadsafe(PendingImageCopyBuffers::default);
        let mut pending = self
            .user_data()
            .get::<PendingImageCopyBuffers>()
            .unwrap()
            .lock()
            .unwrap();
        let was_empty = pending.is_empty();

        let mut dropped_session_cap = 0;
        while pending
            .iter()
            .filter(|(queued_session, _)| queued_session == &session)
            .count()
            >= MAX_PENDING_FRAMES_PER_SESSION
        {
            let Some(index) = pending
                .iter()
                .position(|(queued_session, _)| queued_session == &session)
            else {
                break;
            };
            let (_, dropped_frame) = pending.remove(index);
            dropped_frame.fail(CaptureFailureReason::Unknown);
            dropped_session_cap += 1;
        }

        let mut dropped_output_cap = 0;
        while pending.len() >= MAX_PENDING_FRAMES_PER_OUTPUT {
            let (_, dropped_frame) = pending.remove(0);
            dropped_frame.fail(CaptureFailureReason::Unknown);
            dropped_output_cap += 1;
        }

        pending.push((session, frame));
        note_capture_queue(1, 0, dropped_session_cap, dropped_output_cap, pending.len());
        was_empty
    }
    fn remove_frame(&mut self, frame: &FrameRef) {
        if let Some(pending) = self.user_data().get::<PendingImageCopyBuffers>() {
            pending.lock().unwrap().retain(|(_, f)| f != frame);
        }
    }
    fn take_pending_frames(&self) -> Vec<(SessionRef, Frame)> {
        let frames = self
            .user_data()
            .get::<PendingImageCopyBuffers>()
            .map(|pending| std::mem::take(&mut *pending.lock().unwrap()))
            .unwrap_or_default();
        note_capture_queue(0, frames.len() as u64, 0, 0, 0);
        frames
    }
}

impl SessionHolder for Workspace {
    fn add_session(&mut self, session: Session) {
        self.image_copy.sessions.push(session);
    }

    fn remove_session(&mut self, session: &SessionRef) {
        self.image_copy.sessions.retain(|s| s != session);
    }
    fn sessions(&self) -> Vec<SessionRef> {
        self.image_copy
            .sessions
            .iter()
            .map(|s| (*s).clone())
            .collect()
    }

    fn add_cursor_session(&mut self, session: CursorSession) {
        self.image_copy.cursor_sessions.push(session);
    }

    fn remove_cursor_session(&mut self, session: &CursorSessionRef) {
        self.image_copy.cursor_sessions.retain(|s| s != session);
    }
    fn cursor_sessions(&self) -> Vec<CursorSessionRef> {
        self.image_copy
            .cursor_sessions
            .iter()
            .map(|s| (*s).clone())
            .collect()
    }

    fn mark_capture_active(&self) {
        self.image_copy.mark_capture_active();
    }

    fn is_capture_active(&self) -> bool {
        self.image_copy.is_capture_active()
    }
}

impl SessionHolder for CosmicSurface {
    fn add_session(&mut self, session: Session) {
        self.user_data()
            .insert_if_missing(ImageCopySessionsData::default);
        self.user_data()
            .get::<ImageCopySessionsData>()
            .unwrap()
            .borrow_mut()
            .sessions
            .push(session);
    }

    fn remove_session(&mut self, session: &SessionRef) {
        if let Some(data) = self.user_data().get::<ImageCopySessionsData>() {
            data.borrow_mut().sessions.retain(|s| s != session);
        }
    }
    fn sessions(&self) -> Vec<SessionRef> {
        self.user_data()
            .get::<ImageCopySessionsData>()
            .map_or(Vec::new(), |sessions| {
                sessions
                    .borrow()
                    .sessions
                    .iter()
                    .map(|s| (*s).clone())
                    .collect()
            })
    }

    fn add_cursor_session(&mut self, session: CursorSession) {
        self.user_data()
            .insert_if_missing(ImageCopySessionsData::default);
        self.user_data()
            .get::<ImageCopySessionsData>()
            .unwrap()
            .borrow_mut()
            .cursor_sessions
            .push(session);
    }

    fn remove_cursor_session(&mut self, session: &CursorSessionRef) {
        if let Some(data) = self.user_data().get::<ImageCopySessionsData>() {
            data.borrow_mut().cursor_sessions.retain(|s| s != session);
        }
    }

    fn cursor_sessions(&self) -> Vec<CursorSessionRef> {
        self.user_data()
            .get::<ImageCopySessionsData>()
            .map_or(Vec::new(), |sessions| {
                sessions
                    .borrow()
                    .cursor_sessions
                    .iter()
                    .map(|s| (*s).clone())
                    .collect()
            })
    }

    fn mark_capture_active(&self) {
        if let Some(data) = self.user_data().get::<ImageCopySessionsData>() {
            data.borrow().mark_capture_active();
        }
    }

    fn is_capture_active(&self) -> bool {
        self.user_data()
            .get::<ImageCopySessionsData>()
            .map_or(false, |data| data.borrow().is_capture_active())
    }
}
