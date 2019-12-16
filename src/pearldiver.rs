use crate::constants::*;

use crossbeam::channel::{Receiver, Sender};
use crossbeam::unbounded;

use std::task::Poll;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum State {
    Reset,
    Ready,
    Running,
    Cancelled,
    Completed,
}

pub struct PearlDiver {
    num_cores: usize,
    min_weight_magnitude: usize,
    state: State,
    hi: [u64; STATE_LENGTH],
    lo: [u64; STATE_LENGTH],
}

pub struct SearchFuture {
    nonce_channel: Receiver<Nonce>,
}

pub struct Nonce([i8; NONCE_LENGTH]);

impl PearlDiver {
    pub fn new(num_cores: usize, mwm: usize) -> Self {
        Self {
            num_cores,
            min_weight_magnitude: mwm,
            ..Self::default()
        }
    }

    pub fn prepare(&mut self, input: &[i8; STATE_LENGTH]) {
        if self.state != State::Reset {
            return;
        }
        for i in 0..STATE_LENGTH {
            match input[i] {
                1 => {
                    self.lo[i] = LO_BITS;
                }
                -1 => {
                    self.hi[i] = LO_BITS;
                }
                _ => (),
            }
        }

        self.state = State::Ready;
    }

    pub fn reset(&mut self) {
        match self.state {
            State::Cancelled => (),
            State::Completed => (),
            State::Ready => (),
            _ => return,
        }

        for i in 0..STATE_LENGTH {
            self.hi[i] = HI_BITS;
            self.lo[i] = HI_BITS;
        }

        self.state = State::Reset;
    }

    pub fn sync_search(&mut self) -> Option<Nonce> {
        if self.state != State::Ready {
            return None;
        }

        let (tx, rx) = crossbeam::unbounded::<Nonce>();

        crossbeam::scope(|scope| {
            for _ in 0..self.num_cores {
                let mut lo = [0u64; STATE_LENGTH];
                let mut hi = [0u64; STATE_LENGTH];

                lo.copy_from_slice(&self.lo);
                hi.copy_from_slice(&self.hi);

                scope.spawn(move |_| {
                    //
                    unsafe {
                        transform(&mut lo, &mut hi);
                    }
                });

                base_increment(&mut self.lo, &mut self.hi);
            }
            //
            //Ok(Nonce::default())
            unimplemented!()
        })
        .expect("error executing scope");

        let nonce = rx.recv().expect("error receiving from Nonce channel");

        Some(nonce)
    }

    pub fn async_search(&mut self) -> Option<SearchFuture> {
        if self.state != State::Ready {
            return None;
        }

        // create channel
        let (sender, receiver) = crossbeam::unbounded::<Nonce>();

        self.state = State::Running;

        unimplemented!()
    }

    pub(crate) fn state(&self) -> State {
        self.state
    }
}

fn base_increment(lmid: &mut [u64; STATE_LENGTH], hmid: &mut [u64; STATE_LENGTH]) {
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

fn core_increment(lmid: &mut [u64; STATE_LENGTH], hmid: &mut [u64; STATE_LENGTH]) -> bool {
    let mut fin = CORE_INCR_START;

    for i in CORE_INCR_START..HASH_LENGTH {
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

    fin < HASH_LENGTH
}

unsafe fn transform(lmid: &mut [u64; STATE_LENGTH], hmid: &mut [u64; STATE_LENGTH]) {
    let mut lpad = [0u64; STATE_LENGTH];
    let mut hpad = [0u64; STATE_LENGTH];

    let mut lsrc = lmid.as_mut_ptr();
    let mut hsrc = hmid.as_mut_ptr();

    let mut lpad = lpad.as_mut_ptr();
    let mut hpad = hpad.as_mut_ptr();

    let mut lswp: *mut u64;
    let mut hswp: *mut u64;

    for _ in 0..(NUM_ROUNDS - 1) {
        for j in 0..STATE_LENGTH {
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
    for j in 0..HASH_LENGTH {
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

impl Default for PearlDiver {
    fn default() -> Self {
        Self {
            num_cores: num_cpus::get(),
            min_weight_magnitude: 14,
            state: State::Reset,
            hi: [HI_BITS; STATE_LENGTH],
            lo: [HI_BITS; STATE_LENGTH],
        }
    }
}

impl std::future::Future for SearchFuture {
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
