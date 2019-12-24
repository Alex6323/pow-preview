mod constants;
mod curl;
mod pearldiver;
mod trits;

use crate::constants::{CURL_HASH_LENGTH, NONCE_HASH_POS, NONCE_TX_POS, TRANSACTION_LENGTH};
use crate::curl::Curl;
use crate::pearldiver::Transaction;
use crate::pearldiver::{CoreCount, Difficulty};
use crate::pearldiver::{PearlDiver, PearlDiverState};

const DIFFICULTY: usize = 14;

fn main() {
    // Create some random trits representing a ternary serialized transaction
    let mut random_trits = [0i8; TRANSACTION_LENGTH];
    trits::random_fill(&mut random_trits);

    let transaction = Transaction(random_trits);
    let is_valid = validity_check(&transaction, Difficulty(DIFFICULTY));
    println!("is valid: {}", is_valid);

    // Create a default PearlDiver instance, i.e. try to use 4 cores, and difficulty 14
    let mut pdiver = PearlDiver::new(CoreCount(4), Difficulty(DIFFICULTY));

    // Search for a nonce in synchronous manner. That means that this call
    // will block the main thread until `PearlDiver` completes by finding a nonce or
    // exhausting the whole search space without success.
    pdiver.search_sync(&transaction);

    match pdiver.state() {
        PearlDiverState::Completed(Some(nonce)) => {
            println!("{:?}", nonce.to_vec());

            // Update the transaction with the found nonce
            let mut powed_trits = [0i8; TRANSACTION_LENGTH];
            powed_trits[0..NONCE_TX_POS].copy_from_slice(&(*transaction)[..]);
            powed_trits[NONCE_TX_POS..].copy_from_slice(nonce.as_slice());

            let transaction = Transaction(powed_trits);
            assert!(validity_check(&transaction, Difficulty(DIFFICULTY)));
            println!("is valid: {}", is_valid);
        }
        PearlDiverState::Completed(None) => {
            println!("I'm sorry Dave, I'm afraid I can't do that");
        }
        _ => unreachable!(),
    }

    // Now for the asynchronous execution:
    // What happens now is that instead of blocking we'll receive a `PearlDiverSearch` future immediatedly.
    let search = pdiver.search_async(&transaction);
}

// Checks, if the given transaction is valid in terms of PoW.
fn validity_check(tx_trits: &Transaction, difficulty: Difficulty) -> bool {
    let mut hash = [0i8; CURL_HASH_LENGTH];

    let mut curl = Curl::default();
    curl.reset();
    curl.absorb(&tx_trits[..], 0, tx_trits.len());
    curl.squeeze(&mut hash, 0, CURL_HASH_LENGTH);

    println!("Hash={:?}", hash.to_vec());

    for i in (CURL_HASH_LENGTH - *difficulty)..CURL_HASH_LENGTH {
        if hash[i] != 0 {
            return false;
        }
    }
    true
}
