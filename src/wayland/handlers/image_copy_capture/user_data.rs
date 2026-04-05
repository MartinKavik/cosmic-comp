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
        CursorSession, CursorSessionRef, Frame, FrameRef, Session, SessionRef,
    },
};

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
static MONOTONIC_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn current_time_ms() -> u64 {
    MONOTONIC_EPOCH.elapsed().as_millis() as u64
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
        pending.push((session, frame));
        was_empty
    }
    fn remove_frame(&mut self, frame: &FrameRef) {
        if let Some(pending) = self.user_data().get::<PendingImageCopyBuffers>() {
            pending.lock().unwrap().retain(|(_, f)| f != frame);
        }
    }
    fn take_pending_frames(&self) -> Vec<(SessionRef, Frame)> {
        self.user_data()
            .get::<PendingImageCopyBuffers>()
            .map(|pending| std::mem::take(&mut *pending.lock().unwrap()))
            .unwrap_or_default()
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
