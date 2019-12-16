//! Standard Curl (type) implementation

use crate::constants::CURL_HASH_LENGTH as HASH_LENGTH;
use crate::constants::CURL_STATE_LENGTH as STATE_LENGTH;
use crate::constants::NUM_CURL_ROUNDS as NUM_ROUNDS;

const TRUTH_TABLE: [i8; 11] = [1, 0, -1, 2, 1, -1, 0, 2, -1, 1, 0];

// The standard Sponge Curl implementation.
pub struct Curl {
    num_rounds: usize,
    state: [i8; STATE_LENGTH],
    scratchpad: [i8; STATE_LENGTH],
}

impl Curl {
    pub fn new(num_rounds: usize) -> Self {
        Self {
            num_rounds,
            ..Self::default()
        }
    }

    pub fn absorb(&mut self, trits: &[i8], mut offset: usize, mut length: usize) {
        loop {
            let chunk_length = {
                if length < HASH_LENGTH {
                    length
                } else {
                    HASH_LENGTH
                }
            };

            self.state[0..chunk_length].copy_from_slice(&trits[offset..offset + chunk_length]);

            self.transform();

            offset += chunk_length;

            if length > chunk_length {
                length -= chunk_length;
            } else {
                break;
            }
        }
    }

    pub fn squeeze(&mut self, trits: &mut [i8], mut offset: usize, mut length: usize) {
        loop {
            let chunk_length = {
                if length < HASH_LENGTH {
                    length
                } else {
                    HASH_LENGTH
                }
            };

            trits[offset..offset + chunk_length].copy_from_slice(&self.state[0..chunk_length]);

            self.transform();

            offset += chunk_length;

            if length > chunk_length {
                length -= chunk_length;
            } else {
                break;
            }
        }
    }

    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|t| *t = 0);
    }

    fn transform(&mut self) {
        let mut scratchpad_index = 0;

        for _ in 0..self.num_rounds {
            self.scratchpad.copy_from_slice(&self.state);
            for state_index in 0..STATE_LENGTH {
                let prev_scratchpad_index = scratchpad_index;

                if scratchpad_index < 365 {
                    scratchpad_index += 364;
                } else {
                    scratchpad_index -= 365;
                }

                self.state[state_index] = TRUTH_TABLE[(self.scratchpad[prev_scratchpad_index]
                    + (self.scratchpad[scratchpad_index] << 2)
                    + 5) as usize];
            }
        }
    }
}

impl Default for Curl {
    fn default() -> Self {
        Curl {
            num_rounds: NUM_ROUNDS,
            state: [0; STATE_LENGTH],
            scratchpad: [0; STATE_LENGTH],
        }
    }
}

#[cfg(test)]
mod curl_tests {
    use super::*;
    use crate::convert::*;

    const MAINNET_TRYTES_1: &str = "TLFCFY9IMZVINTAZRCUWTKAFENIBIFOGKWDZQIRTYSVVHTSIIZZ9RLUYVTLXEHACXIUFJJQNFRJYMGGYDWOBNMTPFE9CGVVTREVUJKIXRHSOPFAXMNEMHEW9ZE9HVFEDEORKWGLNECZ9MXLDHPBAOMO9ZMSZJCZLAWWZKOLHBASHYNMCBCPZOXOLLVMFZVCTUDQZSIUSITRDHHXGAOVTOMSKDTZXLSCNHNXJNVGOTZPJDRHOBUAPIAIGLCETVDWSOPEKAOWBNUIEUTTLPFQLRYVRJQJOCBVOZEK9TQMJQUPEZKLHIVMO9TRIUBQNXJYIXFUWFUYWDIIDBQXRYULR9RXPSLTRFY9IIMQBLGOXUZJAKFSEJCSTYP9SWRFCNTMDMRFFWCVZTNFYLFZISPCQ99OSTMJBNLYCQLKWETRLJEOEBJZBO9ZUZMGQIRCCLBANSVYABGKMQCKWIWHHH9FGKGIURCJDKTIQBFENQCYWAX9WHNQ9OKGIWILNFJGMERJNBHDPNFCASDKZLOXLALOSMUFXYKKCDKWVX9PBOVXMAICVTHBLPWPFWJWYBTLJLXNOHREEFTJDLTYPPFMZ9MTPBHNXQL9MXRLGMKRN9EJYZMDZEOZOMKVYWKORKIBKDYZTCPOHYIADIVJWCHRVWCE9DSSHIEEINJYQWBFBTOCZOBL9LLFKWIFAJT9ZQKEUZTBARTEYUBYQOKMRMKWLTJOPVKIDXIUWQFLVWBTAYNOREZSCKAGRGVRLQUBUGKKHLL9YBFMGUMNSUMAXMCRQOQHBYJJRBMFQIUPZEBXFMHYJMAMAHUMMBLRDPBIOMJ9OCHBSBIFX9YSXPPVDMUCICHCSYRWUXXUEROHXGGEJBFJE9S9QGAQ9YOPIZOKGXRXMMFBLGVMC9QXJZTI99TATFJDJORMGJPAQGQICFHYAMWEUKWYYKIGTWYPNC9ZPQEKWAOZVCBIPZUTZUKJXFPWTQUKWIYJBULBJEJZGYEHVYUHFROLQYYPI9WCXHHWEITITPTXMTBWLJRAYV9LZK9FVGBOQRSWEFRMWBKBHAYWETHDTAAPOPPHFOX9PYQAXDVMWXGW9HDTLSINGRWGODCBNVXXYVDKJ9OROIZAULXMZUEVSDPWUJC9FEQAWMDOI9TALZAHX9ZHYSQEJOSZTHZPKWMZBTWUKNJUJNTZRWEYVWUAXVEP9NSZVYHLHZWDDTCQQTCDHTQPZXTM9ERHNNEORYBUKIRJPZORWXJDRRURZCBYLMFZKSZZVJIWXBXSKJMKUAFYKRQKVIGJJGYLXKFWZEIU9JJXRQSOFDLGXELTVBXKPDLKRLJTGVOD9QGIVVWS9EZAMBPDIEABEJJKTYQZVOD9TIGXPDJGJBRLHXCKKFFVQXFPQNKLMOMOJUDNFZCYEP9CQVNQKRYLCMCFNM9JIE9XUCDBX9ABNHZTSRROFYZCXDRLRBMYYRWUEWHC9QGGHBIQVBISISOZWXGXKQWSOASERXWNQXHWUGXDKIVDDWZZIRIERRSEOMEREYYCO9QIXKQOZQZALPBNQCBJWPV9BYDGYTDJPHXFZQ9CQZIDZTORKIABS9LFWOPWISFESVOTWIBTGDFIZBDOAJO9DJVAIQVUYEAWPRETXYWFMMUUUEUMWPGTWEUSZHJUCYGZDCSGVZGNTJBWGHGYZEOTOVIYAODKWJJLJFZGIKVGUYXRGAFMOFDM9SHSWVSDKAJGEVCORATXJHEGLYTVCGCTXZVUFVLZ9CYFCA9MM9STIZHKTGYJUACFVEGSZYJBNRWTRO9JUWZWOSPGJYIRTQSD9EPHONGYDWUQXYRHGXUSVGIAPVGOLLFQTQOYSOMHAOCNVKLPGRKIEVZGCFVWLTBEMM9QMUML9RVYCMOFIUCNTTALZKSGIPVNLFUGDPTHVGKDUIOZMKAEPYSYZTNFTMWJY99VGIM9YHI9WIVVJAANTHPKT9HOWWZSYRDMVJCSKASOZOOPAUOMMSOWNUTTGREQWPQDKRGGSODHKPFUIXKLVDFJSOQH9ZYMREQNXHHPOEISKPGTNIEBKV9SEFTKZZZVXQAYFPYTDMJVUULL9YNMITHTRB9GKILOFJCCYXKMPIYNNOXTVNLDKTODGEADIRIUXHNGVAAIEFYG9BE9BRNAZUABPF9BVODCZGPXBLBVJIXYLLYDVDUKVYGIWETMSKYXGYMXSXGKPDZMG9NOFIMSKFKIHTQSAVGIWERREF9MEAOCDE99FXRR9FDCKOZOJBTOZEVLLCASBONUMPDVD9XWSHEGZ9999999999999999999999999999999999999999999999VPRPPZD99A99999999J99999999KOJZIA9PSFRKG9ZUOJO9PGDIEFPGPSDKVPVBSXDIOOXAPZHKLJHEULIJKYRTDXOJKTRFYYSABGTBRKVCBBZZSWTVHQSQGJKQAHLINBNNLFTQERSITF9BAJCODBNLLQEQZETPQBGWFYCOBUARDAGTCGQCGOUBLA9999QPBMLSSKBO9ILX9QKYCAXNHLK9KFUJYO99GOO99VYROHOVXACRKYPFVY9JRSHJIKFGBHOCXQFPMZZ9999999999999999999999999999999HKJSFUCME999999999MMMMMMMMMCVMNOI9PFCHLRVXSUEOCRLTRMUF";
    const MAINNET_HASH_1: &str =
        "MGPBAHYHKSQMMXXONAOOEDQS9RFEKMOOJUCGXSFYLXBHQFWIHMJGFJWDSZTGKHNBCSENCXSPQOSZ99999";

    #[test]
    fn curl_works() {
        let tx_trits = from_tryte_string(MAINNET_TRYTES_1);
        //println!("{:?}", tx_trits);
        //println!("{}", tx_trits.len());
        let mut curl = Curl::default();
        curl.absorb(&tx_trits, 0, tx_trits.len());

        let mut hash_trits = vec![0i8; HASH_LENGTH];
        curl.squeeze(&mut hash_trits, 0, HASH_LENGTH);

        println!("{:?}", hash_trits);
        println!("{}", hash_trits.len());

        let tryte_string = string_from_trytes(&trytes_from_trits(&hash_trits));
        println!("{}", tryte_string);
        assert_eq!(MAINNET_HASH_1, tryte_string);
    }
}
