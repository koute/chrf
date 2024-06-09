#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

#[cfg(not(feature = "ahash"))]
use std::collections::HashMap;

#[cfg(feature = "ahash")]
use ahash::HashMap;

/// A trait representing a container for ngram counts.
pub trait Ngrams<G = char>: Default
where
    G: Copy + Default,
{
    #[doc(hidden)]
    fn _feed_impl<const N: usize>(&mut self, count: usize, buffer: [G; N]);

    #[doc(hidden)]
    fn _chrf_impl(beta: f64, tl: &Self, refs: &Self) -> (f64, usize);

    /// Adds all of the items from `iter`.
    fn feed_from(&mut self, iter: impl IntoIterator<Item = G>);

    /// Clears all of the ngrams.
    fn clear(&mut self);
}

#[derive(Default, Debug)]
struct N0<G>(core::marker::PhantomData<G>);

impl<G> Ngrams<G> for N0<G>
where
    G: Copy + Default,
{
    fn _feed_impl<const N: usize>(&mut self, _count: usize, _buffer: [G; N]) {}
    fn _chrf_impl(_beta: f64, _tl: &Self, _refs: &Self) -> (f64, usize) {
        (0.0, 0)
    }
    fn feed_from(&mut self, _iter: impl IntoIterator<Item = G>) {}
    fn clear(&mut self) {}
}

macro_rules! impl_ngrams {
    ($(($name:ident = $width:expr, $next:ident))*) => {
        $(
            #[derive(Default, Debug)]
            pub struct $name<G = char> {
                ngrams: HashMap<[G; $width], u32>,
                next: $next<G>,
            }

            const _: () = {
                assert!($width != 0);
            };

            impl From<&str> for $name<char> {
                fn from(text: &str) -> Self {
                    let mut out = Self::default();
                    out.feed(text);
                    out
                }
            }

            impl $name<char> {
                /// Adds all of the ngrams from `text` except spaces.
                fn feed(&mut self, text: &str) {
                    self.feed_from(text.chars().filter(|&ch| ch != ' '))
                }
            }

            impl<G> Ngrams<G> for $name<G> where G: Copy + Default + PartialEq + Eq + core::hash::Hash {
                #[inline(always)]
                fn _feed_impl<const N: usize>(&mut self, count: usize, buffer: [G; N]) {
                    assert!(N >= $width);
                    if count >= $width {
                        let mut ngram = [G::default(); $width];
                        ngram.copy_from_slice(&buffer[buffer.len() - $width..]);
                        *self.ngrams.entry(ngram).or_insert(0) += 1;
                    }
                    self.next._feed_impl(count, buffer);
                }

                #[inline(always)]
                fn _chrf_impl(beta: f64, tl: &Self, refs: &Self) -> (f64, usize) {
                    let mut total_tl = 0;
                    for &count_tl in tl.ngrams.values() {
                        total_tl += count_tl;
                    }

                    let mut matching = 0;
                    let mut total_ref = 0;
                    for (ngram, &count_ref) in &refs.ngrams {
                        total_ref += count_ref;
                        if let Some(&count_tl) = tl.ngrams.get(ngram) {
                            matching += core::cmp::min(count_ref, count_tl);
                        }
                    }

                    let chr_tl = if total_tl > 0 {
                        matching as f64 / total_tl as f64
                    } else {
                        1e-16
                    };

                    let chr_ref = if total_ref > 0 {
                        matching as f64 / total_ref as f64
                    } else {
                        1e-16
                    };

                    let beta2 = beta.powi(2);
                    let numerator = (1.0 + beta2) * (chr_tl * chr_ref);
                    let mut denominator = (beta2 * chr_tl + chr_ref);
                    if denominator < 1e-16 {
                        denominator = 1e-16;
                    }

                    let score = numerator / denominator;
                    let (next_score, next_count) = Ngrams::_chrf_impl(beta, &tl.next, &refs.next);
                    (score + next_score, next_count + 1)
                }

                fn clear(&mut self) {
                    self.ngrams.clear();
                    self.next.clear();
                }

                fn feed_from(&mut self, iter: impl IntoIterator<Item = G>) {
                    let mut ngram = [G::default(); $width];
                    let mut count = 0;
                    for ch in iter {
                        #[allow(clippy::reversed_empty_ranges)]
                        for n in 0..$width - 1 {
                            ngram[n] = ngram[n + 1];
                        }
                        ngram[$width - 1] = ch;
                        count += 1;
                        self._feed_impl(count, ngram);
                    }
                }
            }
        )*
    }
}

impl_ngrams! {
    (N1 = 1, N0)
    (N2 = 2, N1)
    (N3 = 3, N2)
    (N4 = 4, N3)
    (N5 = 5, N4)
    (N6 = 6, N5)
    (N7 = 7, N6)
    (N8 = 8, N7)
    (N9 = 9, N8)
    (N10 = 10, N9)
    (N11 = 11, N10)
    (N12 = 12, N11)
}

/// Calculates a custom chrF score.
///
/// NOTE: Unlike [chrf3] the score returned by this function is *not* multiplied by 100.
pub fn chrf<T>(beta: f64, translation: &T, reference: &T) -> f64
where
    T: Ngrams,
{
    let (sum, count) = Ngrams::_chrf_impl(beta, translation, reference);
    sum / count as f64
}

/// Calculates a chrF3 score.
pub fn chrf3(translation: &N6, reference: &N6) -> f64 {
    chrf(3.0, translation, reference) * 100.0
}

#[test]
fn test_chrf3() {
    {
        let tl = "aoeu33";
        let refs = "axeu33";
        let score = chrf3(&tl.into(), &refs.into());
        assert!(
            (score - 37.7778).abs() < 0.0001,
            "unexpected score: {score} (test 1)"
        );
    }

    {
        let tl = "Recent offers of evacuating residents from the Syrian regime and Russia sound like only thinly veiled threats, pediatricians, surgeons and other doctors have said.";
        let refs = "Recent offers of evacuation form the regime and Russia had sounded like thinly-veiled threats, said the surgeons paediatricians and other doctors.";
        let score = chrf3(&tl.into(), &refs.into());
        assert!(
            (score - 69.8328).abs() < 0.0001,
            "unexpected score: {score} (test 1)"
        );
    }
}
