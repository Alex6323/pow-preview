mod constants;
mod curl;
mod pearldiver;
mod trits;

use crate::constants::{HASH_LENGTH, NONCE_START, TRANSACTION_LENGTH};
use crate::curl::Curl;
use crate::pearldiver::PearlDiver;

fn main() {
    let tx_trits_without_nonce = trits::random(NONCE_START);

    let mut curl = Curl::default();
    curl.absorb(&tx_trits_without_nonce[..]);

    let mut pd = PearlDiver::default();
    pd.prepare(&curl.state());

    // synchronous search
    if let Some(nonce) = pd.sync_search() {
        println!("{:?}", nonce.to_vec());

        let mut tx_trits = [0i8; TRANSACTION_LENGTH];
        tx_trits[0..NONCE_START].copy_from_slice(&tx_trits_without_nonce[..]);
        tx_trits[NONCE_START..].copy_from_slice(nonce.as_slice());

        let mut hash = [0i8; HASH_LENGTH];
        curl.reset();
        curl.absorb(&tx_trits[..]);
        curl.squeeze(&mut hash);

        println!("{:?}", hash.to_vec());
    } else {
        println!("no nonce found");
    }
}
