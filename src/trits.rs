use rand::Rng;

pub(crate) fn random_fill(target: &mut [i8]) {
    target
        .iter_mut()
        .for_each(|v| *v = rand::thread_rng().gen_range(-1, 2));
}
