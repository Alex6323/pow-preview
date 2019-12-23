use crate::constants::*;

use std::ops::Deref;
use std::sync::{Arc, RwLock};
use std::task::Poll;

/// A type to encode 64 Curl states in a compressed way that is optimal for parallel processing.
/// It only requires 2 bits (2^2) to represent a trit, and therefore can store 8 trits per 2 bytes,
/// or 4 trits per byte, rather than 1 trit per byte as the T1B1 encoding.
/// (BCT = Binary encoded ternary)
struct Curl64State {
    hi: [u64; CURL_STATE_LENGTH],
    lo: [u64; CURL_STATE_LENGTH],
}

/// A few type aliases to increase code readability.
type WithCarry = bool;
type Exhausted = bool;

/// The balanced trit representation of a serialized transaction.
pub struct Transaction(pub [i8; TRANSACTION_LENGTH]);

/// The balanced trit representatin of a nonce.
#[derive(Copy)]
pub struct Nonce(pub [i8; NONCE_LENGTH]);

/// New-types that increase readability and type-safety.
#[derive(Clone)]
pub struct CoreCount(pub usize);
#[derive(Clone)]
pub struct Difficulty(pub usize);

/// The various states of `PearlDiver`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum PearlDiverState {
    Created,
    Searching,
    Cancelled,
    Completed(Option<Nonce>),
}

/// The `PearlDiver` abstraction itself.
#[derive(Clone)]
pub struct PearlDiver {
    num_cores: CoreCount,
    difficulty: Difficulty,
    state: Arc<RwLock<PearlDiverState>>,
}

/// The `PearlDiver` search for a nonce is a computation which yields a result at some point in the future.
pub struct PearlDiverSearch {
    pdiver: PearlDiver,
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
    /// NOTE: this function has no return value. Once this function returns extract the nonce by querying
    /// PearlDiver's state.
    pub fn search_sync(&mut self, transaction: &Transaction) {
        assert!(self.state() == PearlDiverState::Created);

        // Create the Curl state with all available immutable information (that is everything but the nonce)
        // We call this the pre-state or in other implementations the mid-state (which is confusing because
        // we have everything absorbed already except for the nonce).
        let mut prestate = create_curl_prestate(transaction);

        self.set_state(PearlDiverState::Searching);

        let num_cores = self.num_cores.clone();

        crossbeam::scope(|scope| {
            for _ in 0..*num_cores {
                // This 'pre-state' will now be copied for each thread we'll spawn, which then owns the copy and can
                // mutate it without interference. In terms of performance those are const-time operations and hence
                // don't really matter.
                let mut state_thr = prestate.clone();
                let pdstate = self.state.clone(); // we need to clone the Arc here directly
                let difficulty = self.difficulty.clone();

                scope.spawn(move |_| {
                    // We do want to have this allocated only once before the hot-loop
                    let mut state_tmp = Curl64State::new(BITS1);

                    // This loop should be as optimized as possible.
                    // Luckily with an `RwLock` reads can happen in parallel.
                    while *pdstate.read().unwrap() == PearlDiverState::Searching {
                        // Do the final Curl transformation yielding the final Curl state
                        unsafe {
                            transform(&mut state_thr, &mut state_tmp);
                        }

                        // Check if the nonce satisfies the difficulty setting
                        if let Some(nonce) = find_nonce(&state_thr, &difficulty) {
                            // Signal to all other threads that we (the current thread) won the race
                            //let mut s = *pdstate.write().unwrap();
                            //s.set_state(PearlDiverState::Completed(Some(nonce)));
                            *pdstate.write().unwrap() = PearlDiverState::Completed(Some(nonce));
                            break;
                        } else {
                            // Select the next nonce, or if that isn't possible break out of the loop, and end this
                            // thread.
                            // NOTE: this extra scope with that additional binding isn't necessary, but good for
                            // readability and optimized away by the compiler anyways (0-cost).
                            if {
                                let exhausted = inner_increment(&mut state_thr);
                                exhausted
                            } {
                                break;
                            }
                        }
                    }
                });

                outer_increment(&mut prestate);
            }
        })
        .unwrap();

        // If we reach this point, but the PearlDiver state hasn't been changed to `Completed`, then we exhausted the
        // whole search space without finding a valid nonce, and we end we switch to `Completed(None)` manually.
        if self.state() == PearlDiverState::Searching {
            self.set_state(PearlDiverState::Completed(None));
        }
    }

    pub fn search_async(&mut self) -> Option<PearlDiverSearch> {
        assert!(self.state() == PearlDiverState::Created);
        self.set_state(PearlDiverState::Searching);
        unimplemented!()
    }

    pub fn cancel(&mut self) {
        if self.state() == PearlDiverState::Searching {
            self.set_state(PearlDiverState::Cancelled);
        }
    }

    pub fn state(&self) -> PearlDiverState {
        *self.state.read().unwrap()
    }

    pub fn set_state(&mut self, state: PearlDiverState) {
        *self.state.write().unwrap() = state;
    }
}

fn outer_increment(prestate: &mut Curl64State) {
    for i in BASE_INCR_START..CORE_INCR_START {
        let with_carry = prestate.bit_add(i);
        if !with_carry {
            break;
        }
    }
}

fn inner_increment(prestate: &mut Curl64State) -> Exhausted {
    // we have not exhausted the search space until each add
    // operation produces a carry
    for i in CORE_INCR_START..CURL_HASH_LENGTH {
        if {
            let with_carry = prestate.bit_add(i);
            !with_carry
        } {
            return false;
        }
    }
    true
}

/// Create the Curl state up until the last chunk (of size 243) which contains the nonce trits
fn create_curl_prestate(transaction: &Transaction) -> Curl64State {
    let mut prestate = Curl64State::new(BITS1);
    let mut tmpstate = Curl64State::new(BITS1);

    let mut offset = 0;

    // Normal Curl application (except for doing the same 64 times in parallel) for the first 7776 trits
    for _ in 0..NUM_PRESTATE_ABSORBS {
        for i in 0..CURL_HASH_LENGTH {
            match (*transaction)[offset] {
                1 => prestate.set(i, BITS1, BITS0),
                -1 => prestate.set(i, BITS0, BITS1),
                _ => (),
            }
            offset += 1;
        }

        unsafe {
            transform(&mut prestate, &mut tmpstate);
        }
    }

    // Now we have to partially absorb the first 162 trits (until the nonce trits start)
    for i in 0..NONCE_HASH_POS {
        match (*transaction)[offset] {
            1 => {
                prestate.set(i, BITS1, BITS0);
            }
            -1 => {
                prestate.set(i, BITS0, BITS1);
            }
            _ => (),
        }
        offset += 1;
    }

    // We want each no overlaps when increasing the 64 nonces in parallel. To ensure that
    // we partition the search space appropriately.
    // 4 trits (3^4 = 81 possible, only 64 required)
    // each nonce starts in existence starts out with one out of 64 4-trit sequences
    // so a first check when hashing an incoming transaction can simply check the first
    // 4 trits of the nonce, and discard any message that doesn't has one the 64 4-trit
    // sequences
    prestate.set(NONCE_HASH_POS + 0, H0, L0);
    prestate.set(NONCE_HASH_POS + 1, H1, L1);
    prestate.set(NONCE_HASH_POS + 2, H2, L2);
    prestate.set(NONCE_HASH_POS + 3, H3, L3);

    prestate
}

/// NOTE: To prevent needless allocations we allocate the scratchpad once outside of this function.
unsafe fn transform(pre: &mut Curl64State, tmp: &mut Curl64State) {
    let (mut hpre, mut lpre) = pre.as_mut_ptr();
    let (mut htmp, mut ltmp) = tmp.as_mut_ptr();

    let mut lswp: *mut u64;
    let mut hswp: *mut u64;

    for _ in 0..(NUM_CURL_ROUNDS - 1) {
        for j in 0..CURL_STATE_LENGTH {
            let index1 = INDICES[j + 0];
            let index2 = INDICES[j + 1];

            let alpha = *lpre.offset(index1);
            let kappa = *hpre.offset(index1);
            let sigma = *lpre.offset(index2);
            let gamma = *hpre.offset(index2);

            let delta = (alpha | !gamma) & (sigma ^ kappa);

            *ltmp.offset(j as isize) = !delta;
            *htmp.offset(j as isize) = (alpha ^ gamma) | delta;
        }

        lswp = lpre;
        hswp = hpre;
        lpre = ltmp;
        hpre = htmp;
        ltmp = lswp;
        htmp = hswp;
    }

    // since we don't compute a new state after that, we can stop after 'HASH_LENGTH' to save some cycles
    for j in 0..CURL_HASH_LENGTH {
        let index1 = INDICES[j + 0];
        let index2 = INDICES[j + 1];

        let alpha = *lpre.offset(index1);
        let kappa = *hpre.offset(index1);
        let sigma = *lpre.offset(index2);
        let gamma = *hpre.offset(index2);

        let delta = (alpha | !gamma) & (sigma ^ kappa);

        *lpre.offset(j as isize) = !delta;
        *hpre.offset(j as isize) = (alpha ^ gamma) | delta;
    }
}

/// Tries to find a valid nonce, that satiisfies the given difficulty. If successful returns its index.
fn find_nonce(state: &Curl64State, difficulty: &Difficulty) -> Option<Nonce> {
    let mut nonce_probe = BITS1;

    for i in (CURL_HASH_LENGTH - difficulty.0)..CURL_HASH_LENGTH {
        nonce_probe &= state.bit_equal(i); //bit_equal means hi=lo=1 (which encodes a zero trit)

        // if 'nonce_test' ever becomes 0, then this means that nonce of the candidates satisfied
        // the difficulty setting
        if nonce_probe == 0 {
            return None;
        }
    }

    // If we haven't returned yet, there is at least one nonce that passes the given difficulty
    for slot in 0..NUM_SLOTS {
        if (nonce_probe >> slot) & 1 != 0 {
            return Some(extract_nonce(&state, slot));
        }
    }

    unreachable!()
}

///
fn extract_nonce(state: &Curl64State, index: usize) -> Nonce {
    let mut nonce = [0; NONCE_LENGTH];
    let mut offset = 0;

    for i in NONCE_HASH_POS..CURL_HASH_LENGTH {
        let (hi, lo) = state.get(i);
        let mask = 1 << index;

        match (hi & mask, lo & mask) {
            (1, 0) => nonce[offset] = 1,
            (0, 1) => nonce[offset] = -1,
            (_, _) => (),
        }
        offset += 1;
    }

    Nonce(nonce)
}

impl Default for PearlDiver {
    fn default() -> Self {
        Self {
            num_cores: CoreCount(num_cpus::get()),
            difficulty: Difficulty(14),
            state: Arc::new(RwLock::new(PearlDiverState::Created)),
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

impl Curl64State {
    pub fn new(init_value: u64) -> Self {
        Self {
            hi: [init_value; CURL_STATE_LENGTH],
            lo: [init_value; CURL_STATE_LENGTH],
        }
    }

    pub fn set(&mut self, index: usize, hi: u64, lo: u64) {
        self.hi[index] = hi;
        self.lo[index] = lo;
    }

    pub fn get(&self, index: usize) -> (u64, u64) {
        (self.hi[index], self.lo[index])
    }

    pub fn bit_add(&mut self, index: usize) -> WithCarry {
        let hi = self.hi[index];
        let lo = self.lo[index];

        self.hi[index] = lo;
        self.lo[index] = hi ^ lo;

        hi & !lo != 0
    }

    pub fn bit_equal(&self, index: usize) -> u64 {
        !(self.hi[index] ^ self.lo[index])
    }

    // dark art
    unsafe fn as_mut_ptr(&mut self) -> (*mut u64, *mut u64) {
        ((&mut self.hi).as_mut_ptr(), (&mut self.lo).as_mut_ptr())
    }
}

impl Clone for Curl64State {
    fn clone(&self) -> Self {
        let mut hi = [0; CURL_STATE_LENGTH];
        let mut lo = [0; CURL_STATE_LENGTH];

        hi.copy_from_slice(&self.hi);
        lo.copy_from_slice(&self.lo);

        Self { hi, lo }
    }
}

impl Clone for Nonce {
    fn clone(&self) -> Self {
        let mut nonce = [0_i8; NONCE_LENGTH];
        nonce.copy_from_slice(&self.0);

        Self(nonce)
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

impl Eq for Nonce {}
impl PartialEq for Nonce {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(&i, &j)| i == j)
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

impl Deref for Transaction {
    type Target = [i8; TRANSACTION_LENGTH];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
