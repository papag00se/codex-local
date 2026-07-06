//! Session-scoped, turn-reset state for the per-tool guards (search-rumination,
//! fetch exact-repeat).
//!
//! In this Rust research vehicle the tool backends are stateless module
//! functions, so we *simulate* per-session state with a map keyed by the
//! harness's `conversation_id` and scoped to the current turn (`sub_id`): a new
//! user turn is a new task, so the state resets. This keeps one session's
//! rumination memory from bleeding into another (sub-agents, forks, or — for the
//! eventual Python Shepherd — concurrent sessions).
//!
//! In Shepherd this state lives on the **session object**; there is no map and no
//! eviction. The map here is a fork wart, not the real design — see the
//! "guard state is session-scoped" note in `docs/spec/shephard.md`.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Max distinct sessions retained before the oldest-inserted is evicted. Bounds
/// memory in a long-lived process that spawns many sub-agents / forks.
const MAX_SESSIONS: usize = 32;

struct Inner<T> {
    map: HashMap<String, (String, T)>, // session -> (turn_id, state)
    order: VecDeque<String>,           // insertion order, for eviction
}

/// A per-session store whose value is reset whenever the turn id changes.
pub struct SessionTurnStore<T> {
    inner: Mutex<Inner<T>>,
}

impl<T: Default> Default for SessionTurnStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default> SessionTurnStore<T> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    /// Run `f` against this session's turn-scoped state. If the stored turn id
    /// differs from `turn` (a new user turn), the state is reset to its default
    /// first — so rumination memory is per-task, not per-session-forever.
    pub fn with<R>(&self, session: &str, turn: &str, f: impl FnOnce(&mut T) -> R) -> R {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.map.contains_key(session) {
            while g.map.len() >= MAX_SESSIONS {
                match g.order.pop_front() {
                    Some(old) => {
                        g.map.remove(&old);
                    }
                    None => break,
                }
            }
            g.order.push_back(session.to_string());
        }
        let entry = g
            .map
            .entry(session.to_string())
            .or_insert_with(|| (turn.to_string(), T::default()));
        if entry.0 != turn {
            *entry = (turn.to_string(), T::default());
        }
        f(&mut entry.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resets_state_when_turn_changes() {
        let store: SessionTurnStore<Vec<u8>> = SessionTurnStore::new();
        store.with("s", "t1", |v| v.push(1));
        store.with("s", "t1", |v| v.push(2));
        assert_eq!(store.with("s", "t1", |v| v.len()), 2);
        // New turn → fresh state.
        assert_eq!(store.with("s", "t2", |v| v.len()), 0);
    }

    #[test]
    fn sessions_are_isolated() {
        let store: SessionTurnStore<Vec<u8>> = SessionTurnStore::new();
        store.with("a", "t", |v| v.push(1));
        store.with("b", "t", |v| v.push(9));
        assert_eq!(store.with("a", "t", |v| v.clone()), vec![1]);
        assert_eq!(store.with("b", "t", |v| v.clone()), vec![9]);
    }
}
