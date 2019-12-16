use crate::constants::{HASH_LENGTH, STATE_LENGTH};

pub struct Curl {
    num_rounds: usize,
    state: CurlState,
}

pub type CurlState = [i8; STATE_LENGTH];
pub type CurlHash = [i8; HASH_LENGTH];

impl Curl {
    pub fn new(num_rounds: usize) -> Self {
        Self { ..Self::default() }
    }

    pub fn absorb(&mut self, input: &[i8]) {
        unimplemented!()
    }

    pub fn squeeze(&mut self, output: &mut CurlHash) {
        unimplemented!()
    }

    pub fn reset(&mut self) {
        for i in 0..STATE_LENGTH {
            self.state[i] = 0;
        }
    }

    pub(crate) fn state(&self) -> &CurlState {
        &self.state
    }
}

impl Default for Curl {
    fn default() -> Self {
        Self {
            num_rounds: 81,
            state: [0i8; STATE_LENGTH],
        }
    }
}
