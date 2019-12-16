use crate::constants::*;

use std::ops::Deref;
use std::sync::{Arc, RwLock};
use std::task::Poll;

// internal conveniance alias
type BCTStateBits = [u64; CURL_STATE_LENGTH];

pub struct Input(pub [i8; TRANSACTION_LENGTH]);
pub struct Nonce(pub [i8; NONCE_LENGTH]);

#[derive(Clone)]
struct CoreCount(usize);

#[derive(Clone)]
struct Difficulty(usize);

#[derive(Eq, PartialEq)]
pub enum PearlDiverState {
    Created,
    Searching,
    Cancelled,
    Completed,
}

#[derive(Clone)]
pub struct PearlDiver {
    num_cores: CoreCount,
    difficulty: Difficulty,
    state: Arc<RwLock<PearlDiverState>>,
}

pub struct PearlDiverSearch {
    diver: PearlDiver,
}

impl PearlDiver {
    /// Creates a new `PearlDiver` instance.
    pub fn new(num_cores: CoreCount, difficulty: Difficulty) -> Self {
        Self {
            num_cores,
            difficulty,
            ..Self::default()
        }
    }

    /// Searches for a nonce in synchronous/blocking manner.
    pub fn search_sync(&self, input: &Input) -> Option<Nonce> {
        assert!(self.state() == PearlDiverState::Created);

        //
        let (h_prestate, l_prestate) = create_bct_prestate(input);

        // Start searching
        self.set_state(PearlDiverState::Searching);

        crossbeam::scope(|scope| {
            for _ in 0..*self.num_cores {
                scope.spawn(move |_| {
                    // Copy
                    let mut hcpy = [0u64; CURL_STATE_LENGTH];
                    let mut lcpy = [0u64; CURL_STATE_LENGTH];
                    hcpy.copy_from_slice(&h_prestate);
                    lcpy.copy_from_slice(&l_prestate);

                    loop {
                        //
                        unsafe {
                            transform(&mut lcpy, &mut hcpy);
                        }

                        if !running {
                            break;
                        }
                        if let Some(nonce_index) = check(&lo, &hi, self.difficulty) {
                            // nonce found
                            // extract nonce
                            // send it via the channel
                            // send kill signal
                        } else {
                            core_increment(&mut lo, &mut hi);
                        }
                    }
                });

                base_increment(&mut self.lo, &mut self.hi);
            }
        })
        .expect("error executing scope");

        let nonce = receiver.recv().expect("error receiving from Nonce channel");

        Some(nonce)
    }

    pub fn search_async(&mut self) -> Option<PearlDiverSearch> {
        assert!(self.state() == PearlDiverState::Created);

        self.set_state(PearlDiverState::Searching);

        unimplemented!()
    }

    /*
    /// Prepare PearlDiver by providing a full Curl state in one-trit-per-byte (T1B1) encoding.
    /// Note, that for an IOTA transaction (8019 trits) the Nonce is stored in the last 81 trits.
    fn into_bct(input: &Input) -> ([u64; TRANSACTION_LENGTH], [u64; TRANSACTION_LENGTH]) {
        let mut hi = [BITS1; TRANSACTION_LENGTH];
        let mut lo = [BITS1; TRANSACTION_LENGTH];

        for i in 0..CURL_STATE_LENGTH {
            match input[i] {
                1 => {
                    lo[i] = BITS0;
                }
                -1 => {
                    hi[i] = BITS0;
                }
                _ => (),
            }
        }

        (hi, lo)
    }
    */

    pub fn cancel(&mut self) {
        if self.state() == PearlDiverState::Searching {
            self.set_state(PearlDiverState::Cancelled);
        }
    }

    pub fn state(&self) -> PearlDiverState {
        *self.state.read().unwrap()
    }

    fn set_state(&mut self, state: PearlDiverState) {
        *self.state.write().unwrap() = state;
    }

    fn core_loop(
        lmid: &mut BCTStateBits,
        hmid: &mut BCTStateBits,
        mwm: usize,
        stop_sig: bool,
    ) -> Option<Nonce> {
        let mut lcpy = [BITS1; CURL_STATE_LENGTH];
        let mut hcpy = [BITS1; CURL_STATE_LENGTH];

        let mut space_exhausted = false;

        loop {
            // break on search space exhausted or termination signal
            if stop_sig || space_exhausted {
                return None;
            }

            space_exhausted = !core_increment(lmid, hmid);

            lcpy[..].copy_from_slice(&lmid[..]);
            hcpy[..].copy_from_slice(&hmid[..]);

            unsafe {
                transform(&mut lcpy, &mut hcpy);
            }

            if let Some(nonce_index) = check(&lcpy, &hcpy, mwm) {
                let nonce = extract(lmid, hmid, nonce_index);
                return Some(nonce);
            }
        }
    }
}

fn base_increment(lmid: &mut BCTStateBits, hmid: &mut BCTStateBits) {
    for i in BASE_INCR_START..CORE_INCR_START {
        let lo = lmid[i];
        let hi = hmid[i];

        lmid[i] = hi ^ lo;
        hmid[i] = lo;

        let carry = hi & !lo;
        if carry == 0 {
            break;
        }
    }
}

fn core_increment(lmid: &mut BCTStateBits, hmid: &mut BCTStateBits) -> bool {
    let mut fin = CORE_INCR_START;

    for i in CORE_INCR_START..CURL_HASH_LENGTH {
        let lo = lmid[i];
        let hi = hmid[i];

        lmid[i] = hi ^ lo;
        hmid[i] = lo;

        let carry = hi & !lo;
        if carry == 0 {
            break;
        }

        fin += 1;
    }

    fin < CURL_HASH_LENGTH
}

/// Create the Curl state up until the last chunk (of size 243) which contains the nonce trits
fn create_prestate(input: &Input) -> (BCTStateBits, BCTStateBits) {
    let mut h_prestate = [BITS1; CURL_STATE_LENGTH];
    let mut l_prestate = [BITS1; CURL_STATE_LENGTH];
    let mut h_scratchpad = [BITS1; CURL_STATE_LENGTH];
    let mut l_scratchpad = [BITS1; CURL_STATE_LENGTH];

    let mut offset = 0;

    for _ in 0..NUM_PRENONCE_ABSORBS {
        for i in 0..CURL_HASH_LENGTH {
            match input[offset] {
                1 => {
                    l_prestate[i] = BITS0;
                }
                -1 => {
                    h_prestate[i] = BITS0;
                }
                _ => (),
            }
            offset += 1;
        }

        transform(
            &mut h_prestate,
            &mut l_prestate,
            &mut h_scratchpad,
            &mut l_scratchpad,
        );
    }

    for i in 0..NONCE_CURL_POS {
        match input[offset] {
            0 => {
                mid_state_low[i] = HIGH_BITS;
                mid_state_high[i] = HIGH_BITS;
            }
            1 => {
                mid_state_low[i] = LOW_BITS;
                mid_state_high[i] = HIGH_BITS;
            }
            _ => {
                mid_state_low[i] = HIGH_BITS;
                mid_state_high[i] = LOW_BITS;
            }
        }
        offset += 1;
    }

    l_prestate[162 + 0] = L0;
    h_prestate[162 + 0] = H0;
    l_prestate[162 + 1] = L1;
    h_prestate[162 + 1] = H1;
    l_prestate[162 + 2] = L2;
    h_prestate[162 + 2] = H2;
    l_prestate[162 + 3] = L3;
    h_prestate[162 + 3] = H3;
}

unsafe fn transform(
    lpre: &mut BCTStateBits,
    hpre: &mut BCTStateBits,
    lpad: &mut BCTStateBits,
    hpad: &mut BCTStateBits,
) {
    let mut lpre = lpre.as_mut_ptr();
    let mut hpre = hpre.as_mut_ptr();
    let mut lpad = lpad.as_mut_ptr();
    let mut hpad = hpad.as_mut_ptr();

    let mut lswp: *mut u64;
    let mut hswp: *mut u64;

    for _ in 0..(NUM_CURL_ROUNDS - 1) {
        for j in 0..CURL_STATE_LENGTH {
            let index1 = INDICES[j + 0];
            let index2 = INDICES[j + 1];

            let alpha = *lsrc.offset(index1);
            let kappa = *hsrc.offset(index1);
            let sigma = *lsrc.offset(index2);
            let gamma = *hsrc.offset(index2);

            let delta = (alpha | !gamma) & (sigma ^ kappa);

            *lpad.offset(j as isize) = !delta;
            *hpad.offset(j as isize) = (alpha ^ gamma) | delta;
        }

        lswp = lsrc;
        hswp = hsrc;
        lsrc = lpad;
        hsrc = hpad;
        lpad = lswp;
        hpad = hswp;
    }

    // since we don't compute a new state aftere that, we can stop after 'HASH_LENGTH'
    for j in 0..CURL_HASH_LENGTH {
        let index1 = INDICES[j + 0];
        let index2 = INDICES[j + 1];

        let alpha = *lsrc.offset(index1);
        let kappa = *hsrc.offset(index1);

        let sigma = *lsrc.offset(index2);
        let gamma = *hsrc.offset(index2);

        let delta = (alpha | !gamma) & (sigma ^ kappa);

        *lpad.offset(j as isize) = !delta;
        *hpad.offset(j as isize) = (alpha ^ gamma) | delta;
    }
}

fn check(lo: &BCTLane, hi: &BCTLane, mwm: usize) -> Option<usize> {
    let mut nonce_probe = HI_BITS;
    for i in (CURL_HASH_LENGTH - mwm)..CURL_HASH_LENGTH {
        nonce_probe &= !(lo[i] ^ hi[i]);
        if nonce_probe == 0 {
            return None;
        }
    }

    for i in 0..64 {
        if (nonce_probe >> i) & 0x1 != 0 {
            return Some(i);
        }
    }

    unreachable!()
}

impl Default for PearlDiver {
    fn default() -> Self {
        Self {
            num_cores: CoreCount(num_cpus::get()),
            difficulty: Difficulty(14),
            state: PearlDiverState::Initialized,
        }
    }
}

impl std::future::Future for PearlDiverSearch {
    type Output = Option<Nonce>;

    fn poll(self: std::pin::Pin<&mut Self>, _: &mut std::task::Context) -> Poll<Self::Output> {
        /*
        match self.pearldiver.state() {
            State::Cancelled => return Poll::Ready(None),
            State::Completed => return Poll::Ready(Some(self.pearldiver.nonce())),
            _ => return Poll::Pending,
        }
        */
        unimplemented!()
    }
}

impl Nonce {
    pub fn to_vec(&self) -> Vec<i8> {
        self.0.to_vec()
    }

    pub fn as_slice(&self) -> &[i8] {
        &self.0[..]
    }
}

impl Default for Nonce {
    fn default() -> Self {
        Self([0i8; NONCE_LENGTH])
    }
}

impl From<usize> for CoreCount {
    fn from(num_cores: usize) -> Self {
        let max_cores = num_cpus::get();
        if num_cores > max_cores {
            Self(max_cores)
        } else {
            Self(num_cores)
        }
    }
}

impl Deref for CoreCount {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Difficulty {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<usize> for Difficulty {
    fn from(difficulty: usize) -> Self {
        let max_difficulty = CURL_HASH_LENGTH;
        if difficulty > max_difficulty {
            Self(max_difficulty)
        } else {
            Self(difficulty)
        }
    }
}

impl Deref for Input {
    type Target = [i8; TRANSACTION_LENGTH];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
