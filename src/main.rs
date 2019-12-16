mod constants;
mod curl;
mod pearldiver;
mod trits;

use crate::constants::{CURL_HASH_LENGTH, NONCE_START, TRANSACTION_LENGTH};
use crate::curl::Curl;
use crate::pearldiver::Input;
use crate::pearldiver::PearlDiver;

fn main() {
    // Create some random trits representing a ternary serialized transaction
    let mut trits = [0i8; TRANSACTION_LENGTH];
    trits::random_fill(&mut trits);

    let input = Input(trits);

    // Create a default PearlDiver instance, i.e. use all available cores, and difficulty 14
    let mut pdiver = PearlDiver::default();

    // Search for a nonce in synchronous/blocking manner
    if let Ok(Some(nonce)) = pdiver.search_sync(&input) {
        println!("{:?}", nonce.to_vec());

        // Check, if the nonce is valid
        let mut tx_trits = [0i8; TRANSACTION_LENGTH];
        tx_trits[0..NONCE_START].copy_from_slice(&(*input)[..]);
        tx_trits[NONCE_START..].copy_from_slice(nonce.as_slice());

        let mut hash = [0i8; CURL_HASH_LENGTH];
        let mut curl = Curl::default();
        curl.reset();
        curl.absorb(&tx_trits[..], 0, tx_trits.len());
        curl.squeeze(&mut hash, 0, CURL_HASH_LENGTH);

        println!("{:?}", hash.to_vec());
    } else {
        println!("no nonce found");
    }
}
