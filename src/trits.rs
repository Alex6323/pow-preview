use rand::Rng;

pub(crate) fn random(count: usize) -> Vec<i8> {
    (0..count)
        .map(|_| rand::thread_rng().gen_range(-1, 2))
        .collect::<Vec<i8>>()
}
