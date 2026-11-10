//! Rust backend for the two hot stages of `FuzzyPhraseSearcher.find_matches`:
//! the skipgram scan (`Searcher.find_skipgram_matches`) and candidate building
//! (`skip_match.get_skipmatch_candidates` with `candidate_match.CandidatePartial`).
//!
//! The behaviour mirrors the Python code step by step, including its quirks, so that
//! the regression fixture can be matched exactly. Offsets are Unicode code points, as
//! in Python string slicing. All phrase data is computed in Python and passed in once.

#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

mod postprocess;
use postprocess::{adjust_end, adjust_start, char_overlap_ratio, levenshtein_ratio, levenshtein_ratio_with,
                  ngram_overlap_ratio, slice, PatternMasks};

/// A skipgram string packed into an integer: a leading 1 bit, then 21 bits per char.
/// Fits up to 6 characters, so ngram sizes up to 6 are supported.
type Key = u128;
const MAX_NGRAM_SIZE: usize = 6;

fn key_of_chars<I: IntoIterator<Item = char>>(chars: I) -> Key {
    let mut key: Key = 1;
    for c in chars {
        key = (key << 21) | c as Key;
    }
    key
}

fn key_of_str(s: &str) -> Key {
    key_of_chars(s.chars())
}

#[derive(Clone, Copy, Debug)]
struct TextSkip {
    start: i64,
    length: i64,
    end_offset: i64,
    key: Key,
}

/// All skipgrams of a text, in the order of `text2skipgrams`.
fn text_skipgrams(chars: &[char], ngram_size: usize, skip_size: usize, combos: &[Vec<usize>]) -> Vec<TextSkip> {
    let n = chars.len() as i64;
    let mut out = Vec::new();
    if ngram_size == 1 {
        for (ci, &c) in chars.iter().enumerate() {
            out.push(TextSkip { start: 0, length: 1, end_offset: n - ci as i64 + 1, key: key_of_chars([c]) });
        }
        return out;
    }
    if chars.len() <= ngram_size {
        out.push(TextSkip { start: 0, length: n, end_offset: 0, key: key_of_chars(chars.iter().copied()) });
        return out;
    }
    let window_size = ngram_size + skip_size;
    out.reserve(chars.len() * combos.len());
    for start in 0..chars.len() - 1 {
        let window = &chars[start..(start + window_size).min(chars.len())];
        let end_offset = n - start as i64 + 1;
        for combo in combos {
            let last = *combo.last().unwrap();
            if last >= window.len() {
                continue;
            }
            let mut key: Key = (1 << 21) | window[0] as Key;
            for &i in combo {
                key = (key << 21) | window[i] as Key;
            }
            out.push(TextSkip { start: start as i64, length: last as i64 + 1, end_offset, key });
        }
    }
    out
}

/// itertools.combinations(range(1, n), r) in the same (lexicographic) order.
fn combinations(n: usize, r: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(r);
    fn rec(start: usize, n: usize, r: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == r {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            rec(i + 1, n, r, current, out);
            current.pop();
        }
    }
    rec(1, n, r, &mut current, &mut out);
    out
}

struct PhraseData {
    phrase_len: i64,
    skipgram_set_size: usize,
    num_skipgrams: i64,
    late_threshold: i64,
    max_start_offset: i64,
    max_start_end: i64,
    max_end_offset: i64,
    max_end_end: i64,
    /// dense index of each skipgram string of the phrase
    local: FxHashMap<Key, u32>,
    /// per local index: start offset of the first occurrence in the phrase (skipgram_index[s][0])
    first_offset: Vec<i64>,
    freq: Vec<i64>,
    early: Vec<bool>,
    late: Vec<bool>,
}

impl PhraseData {
    fn from_tuple(t: PhraseTuple) -> Result<Self, String> {
        let mut local: FxHashMap<Key, u32> = FxHashMap::default();
        let mut first_offset = Vec::new();
        for (s, o) in &t.8 {
            local.insert(key_of_str(s), first_offset.len() as u32);
            first_offset.push(*o);
        }
        let n = first_offset.len();
        let lookup = |s: &str| -> Result<usize, String> {
            local.get(&key_of_str(s)).map(|&i| i as usize)
                .ok_or_else(|| format!("skipgram {s:?} is not in the phrase's skipgram index"))
        };
        let mut freq = vec![0i64; n];
        for (s, f) in &t.9 {
            freq[lookup(s)?] = *f;
        }
        let mut early = vec![false; n];
        for s in &t.10 {
            early[lookup(s)?] = true;
        }
        let mut late = vec![false; n];
        for s in &t.11 {
            late[lookup(s)?] = true;
        }
        Ok(PhraseData {
            phrase_len: t.0,
            skipgram_set_size: t.1,
            num_skipgrams: t.2,
            late_threshold: t.3,
            max_start_offset: t.4,
            max_start_end: t.5,
            max_end_offset: t.6,
            max_end_end: t.7,
            local,
            first_offset,
            freq,
            early,
            late,
        })
    }
}

/// A text skipgram matched to a phrase: offsets and the skipgram's dense index in that phrase.
#[derive(Clone, Copy, Debug)]
struct MSkip {
    start: i64,
    length: i64,
    local: u32,
}

/// Mirror of candidate_match.CandidatePartial and the functions operating on it.
struct Partial<'a> {
    p: &'a PhraseData,
    list: VecDeque<MSkip>,
    count: Vec<i64>,     // like Counter: per local index, keys stay present at count zero
    in_count: Vec<bool>, // whether the key is present in the Counter
    present: Vec<u32>,   // keys present in the Counter, for iteration and reset
    distinct: usize,     // size of skipgram_set: keys with count > 0
    start: Option<i64>,  // None stands for both -1 and None in Python
    end: i64,
    max_length: i64,
}

impl<'a> Partial<'a> {
    fn new(p: &'a PhraseData, max_length_variance: i64) -> Self {
        Partial {
            p,
            list: VecDeque::new(),
            count: vec![0; p.first_offset.len()],
            in_count: vec![false; p.first_offset.len()],
            present: Vec::new(),
            distinct: 0,
            start: None,
            end: -1,
            max_length: p.phrase_len + max_length_variance,
        }
    }

    /// Same as replacing it with Partial::new, without reallocating.
    fn reset(&mut self) {
        for &k in &self.present {
            self.count[k as usize] = 0;
            self.in_count[k as usize] = false;
        }
        self.present.clear();
        self.list.clear();
        self.distinct = 0;
        self.start = None;
        self.end = -1;
    }

    fn match_start(&self) -> Option<i64> {
        let first = self.list.front()?;
        let offset = first.start - self.p.first_offset[first.local as usize];
        Some(if offset < 0 { 0 } else { offset })
    }

    fn length(&self) -> i64 {
        match self.start {
            None => 0,
            Some(start) => self.end - start,
        }
    }

    fn remove_first(&mut self) {
        let first = self.list.pop_front().unwrap();
        let count = &mut self.count[first.local as usize];
        *count -= 1;
        if *count == 0 {
            self.distinct -= 1;
        }
    }

    fn add(&mut self, skip: MSkip) {
        if self.list.is_empty() && !self.p.early[skip.local as usize] {
            return;
        }
        self.list.push_back(skip);
        if self.start.is_none() {
            self.start = self.match_start();
        }
        if skip.start + skip.length > self.end {
            self.end = skip.start + skip.length;
        }
        let k = skip.local as usize;
        if !self.in_count[k] {
            self.in_count[k] = true;
            self.present.push(skip.local);
        }
        if self.count[k] == 0 {
            self.distinct += 1;
        }
        self.count[k] += 1;
        while self.length() > self.max_length && !self.list.is_empty() {
            self.remove_first();
            self.start = self.match_start();
        }
        while !self.list.is_empty() && !self.p.early[self.list[0].local as usize] {
            self.remove_first();
            self.start = self.match_start();
        }
    }

    #[allow(clippy::if_same_then_else)] // kept literal to match candidate_match.is_match
    fn is_match(&self, skipgram_threshold: f64, max_length_variance: i64) -> bool {
        let Some(first) = self.list.front() else { return false };
        if !self.p.early[first.local as usize] {
            return false;
        }
        let match_len = self.length();
        if match_len > self.p.phrase_len + max_length_variance {
            return false;
        } else if match_len < self.p.late_threshold - max_length_variance {
            return false;
        }
        if !self.p.late[self.list.back().unwrap().local as usize] {
            return false;
        }
        (self.distinct as f64 / self.p.skipgram_set_size as f64) >= skipgram_threshold
    }

    fn shift_start_skip(&mut self) -> bool {
        let length = self.length();
        if length <= self.p.phrase_len {
            return false;
        }
        let start_skip = self.list[0];
        let mut best_phrase_offset = self.p.first_offset[start_skip.local as usize];
        let mut best_index = 0;
        let mut best_start = start_skip.start;
        for (si, skip) in self.list.iter().enumerate() {
            let skip_phrase_offset = self.p.first_offset[skip.local as usize];
            if skip.start - start_skip.start > length - self.p.phrase_len {
                break;
            }
            if skip.start > best_start && skip_phrase_offset <= best_phrase_offset {
                best_index = si;
                best_start = skip.start;
                best_phrase_offset = skip_phrase_offset;
            }
            if !self.p.early[skip.local as usize] {
                break;
            }
        }
        for _ in 0..best_index {
            self.remove_first();
        }
        self.start = self.match_start();
        best_index > 0
    }

    fn count_overlap(&self) -> f64 {
        let mut diff = 0i64;
        let mut total = 0i64;
        for &k in &self.present {
            let count = self.count[k as usize];
            diff += (count - self.p.freq[k as usize]).abs();
            total += count;
        }
        (total - diff) as f64 / self.p.num_skipgrams as f64
    }

    fn same_as(&self, candidate: &Candidate) -> bool {
        self.start == Some(candidate.1) && self.end == candidate.2
    }

    fn to_candidate(&self, phrase_id: u32) -> Result<Candidate, String> {
        let start = self.start.expect("matching candidate has a start offset");
        check_match_string(start, self.end)?;
        Ok((phrase_id, start, self.end, self.count_overlap()))
    }
}

fn check_match_string(start: i64, end: i64) -> Result<(), String> {
    if start == end {
        Err("start and end offset cannot be the same".to_string())
    } else {
        Ok(())
    }
}

/// (phrase id, match start offset, match end offset, skipgram count overlap)
type Candidate = (u32, i64, i64, f64);

/// Candidate plus (char overlap, ngram overlap, Levenshtein similarity)
type ScoredCandidate = (u32, i64, i64, f64, f64, f64, f64);

/// A known word from exact matching: start, end, phrase ids excluded from the main index
/// and phrase ids excluded from the variant index while inside the word.
type KnownWord = (i64, i64, Vec<u32>, Vec<u32>);

#[pyclass(frozen)]
struct SkipgramMatcher {
    ngram_size: usize,
    skip_size: usize,
    max_length_variance: i64,
    skipgram_threshold: f64,
    combos: Vec<Vec<usize>>,
    phrases: Vec<PhraseData>,
    phrase_chars: Vec<Vec<char>>,
    /// bit masks for fast Levenshtein ratios against each phrase (None for phrases over 64 chars)
    phrase_masks: Vec<Option<PatternMasks>>,
    /// lowercased phrase strings (Python's str.lower()) and their masks, for scoring with ignorecase
    phrase_lower_chars: Vec<Vec<char>>,
    phrase_lower_masks: Vec<Option<PatternMasks>>,
    /// main phrase group of each phrase (index into first appearance order is computed per text)
    group: Vec<u32>,
    /// whether the phrase is the main phrase of its group (tie-break in the overlap filter)
    is_main: Vec<bool>,
    /// text skipgram -> (phrase id, the skipgram's dense index in that phrase)
    main_index: FxHashMap<Key, Vec<(u32, u32)>>,
    variant_index: FxHashMap<Key, Vec<(u32, u32)>>,
}

/// Phrase data as passed from Python, see fuzzy_search/search/rust_backend.py.
type PhraseTuple = (
    i64, usize, i64, i64, i64, i64, i64, i64,
    Vec<(String, i64)>, Vec<(String, i64)>, Vec<String>, Vec<String>,
);

#[pymethods]
impl SkipgramMatcher {
    #[new]
    fn new(
        ngram_size: usize,
        skip_size: usize,
        max_length_variance: i64,
        skipgram_threshold: f64,
        phrases: Vec<PhraseTuple>,
        main_index: Vec<(String, Vec<u32>)>,
        variant_index: Vec<(String, Vec<u32>)>,
        phrase_strings: Vec<String>,
        group: Vec<u32>,
        is_main: Vec<bool>,
        phrase_strings_lower: Vec<String>,
    ) -> PyResult<Self> {
        if phrase_strings.len() != phrases.len() || group.len() != phrases.len() || is_main.len() != phrases.len()
            || phrase_strings_lower.len() != phrases.len() {
            return Err(PyValueError::new_err("phrase_strings, group and is_main must have one entry per phrase"));
        }
        if ngram_size == 0 || ngram_size > MAX_NGRAM_SIZE {
            return Err(PyValueError::new_err(format!("ngram_size must be between 1 and {MAX_NGRAM_SIZE}")));
        }
        let phrases = phrases
            .into_iter()
            .map(PhraseData::from_tuple)
            .collect::<Result<Vec<_>, String>>()
            .map_err(PyValueError::new_err)?;
        let build = |index: Vec<(String, Vec<u32>)>| -> PyResult<FxHashMap<Key, Vec<(u32, u32)>>> {
            let mut map: FxHashMap<Key, Vec<(u32, u32)>> = FxHashMap::default();
            for (s, ids) in index {
                let key = key_of_str(&s);
                for id in ids {
                    let p = phrases.get(id as usize).ok_or_else(|| PyValueError::new_err("phrase id out of range"))?;
                    let local = *p.local.get(&key).ok_or_else(|| {
                        PyValueError::new_err(format!("indexed skipgram {s:?} is not a skipgram of phrase {id}"))
                    })?;
                    map.entry(key).or_default().push((id, local));
                }
            }
            Ok(map)
        };
        let main_index = build(main_index)?;
        let variant_index = build(variant_index)?;
        Ok(SkipgramMatcher {
            ngram_size,
            skip_size,
            max_length_variance,
            skipgram_threshold,
            combos: combinations(ngram_size + skip_size, ngram_size - 1),
            phrases,
            phrase_masks: phrase_strings.iter().map(|s| PatternMasks::new(&s.chars().collect::<Vec<_>>())).collect(),
            phrase_chars: phrase_strings.iter().map(|s| s.chars().collect()).collect(),
            phrase_lower_masks: phrase_strings_lower.iter().map(|s| PatternMasks::new(&s.chars().collect::<Vec<_>>())).collect(),
            phrase_lower_chars: phrase_strings_lower.iter().map(|s| s.chars().collect()).collect(),
            group,
            is_main,
            main_index,
            variant_index,
        })
    }

    /// Candidates for one text, in the order Python's get_skipmatch_candidates produces them
    /// (phrases by id, which Python assigns in phrase_string order), and the group ids in the
    /// order Python's result dict has them.
    #[pyo3(signature = (text, known_words, include_variants))]
    fn candidates(&self, py: Python<'_>, text: &str, known_words: Vec<KnownWord>,
                  include_variants: bool) -> PyResult<(Vec<Candidate>, Vec<u32>)> {
        let chars: Vec<char> = text.chars().collect();
        py.detach(|| self.candidates_for(&chars, &known_words, include_variants))
            .map_err(PyValueError::new_err)
    }

    /// Candidates for many texts, computed on `threads` threads (0: rayon's default) without the GIL.
    #[pyo3(signature = (texts, include_variants, threads = 0))]
    fn candidates_batch(&self, py: Python<'_>, texts: Vec<(String, Vec<KnownWord>)>, include_variants: bool,
                        threads: usize) -> PyResult<Vec<Vec<Candidate>>> {
        let run = || {
            texts
                .par_iter()
                .map(|(text, known)| {
                    let chars: Vec<char> = text.chars().collect();
                    self.candidates_for(&chars, known, include_variants).map(|(c, _)| c)
                })
                .collect::<Result<Vec<_>, String>>()
        };
        py.detach(|| {
            if threads == 0 {
                run()
            } else {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| e.to_string())?
                    .install(run)
            }
        })
        .map_err(PyValueError::new_err)
    }

    /// Candidates after word-boundary adjustment and per-phrase overlap filtering, as returned by
    /// FuzzyPhraseSearcher.filter_phrase_candidates. `text` is the original text, `scan_text` the
    /// text the skipgrams are taken from (lowercased when ignorecase).
    #[pyo3(signature = (text, scan_text, known_words, include_variants, use_word_boundaries, punctuation))]
    fn filtered_candidates(&self, py: Python<'_>, text: &str, scan_text: &str, known_words: Vec<KnownWord>,
                           include_variants: bool, use_word_boundaries: bool,
                           punctuation: &str) -> PyResult<Vec<Candidate>> {
        let orig: Vec<char> = text.chars().collect();
        let scan: Vec<char> = scan_text.chars().collect();
        let punct: Vec<char> = punctuation.chars().collect();
        py.detach(|| self.filtered_for(&orig, &scan, &known_words, include_variants, use_word_boundaries, &punct))
            .map_err(PyValueError::new_err)
    }

    /// filtered_candidates for many (text, scan_text, known_words) items on `threads` threads
    /// (0: one per CPU), without holding the GIL.
    #[pyo3(signature = (items, include_variants, use_word_boundaries, punctuation, threads = 0))]
    fn filtered_candidates_batch(&self, py: Python<'_>, items: Vec<(String, String, Vec<KnownWord>)>,
                                 include_variants: bool, use_word_boundaries: bool, punctuation: &str,
                                 threads: usize) -> PyResult<Vec<Vec<Candidate>>> {
        let punct: Vec<char> = punctuation.chars().collect();
        let run = || {
            items
                .par_iter()
                .map(|(text, scan_text, known)| {
                    let orig: Vec<char> = text.chars().collect();
                    let scan: Vec<char> = if scan_text == text { orig.clone() } else { scan_text.chars().collect() };
                    self.filtered_for(&orig, &scan, known, include_variants, use_word_boundaries, &punct)
                })
                .collect::<Result<Vec<_>, String>>()
        };
        py.detach(|| {
            if threads == 0 {
                run()
            } else {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| e.to_string())?
                    .install(run)
            }
        })
        .map_err(PyValueError::new_err)
    }

    /// filtered_candidates plus the match scores (char overlap, ngram overlap, Levenshtein ratio)
    /// between each candidate's phrase string and match string, as PhraseMatch.add_scores computes
    /// them when ignorecase is off.
    ///
    /// With `lower_text` (the text lowercased by Python, with the same length as `text`), the skipgram
    /// scan uses it and the scores compare lowercased phrase strings with slices of it, as with
    /// ignorecase. Python only passes it when lowercasing slices equals slicing the lowercased text.
    #[pyo3(signature = (text, known_words, include_variants, use_word_boundaries, punctuation, lower_text = None))]
    fn scored_candidates(&self, py: Python<'_>, text: &str, known_words: Vec<KnownWord>, include_variants: bool,
                         use_word_boundaries: bool, punctuation: &str,
                         lower_text: Option<&str>) -> PyResult<Vec<ScoredCandidate>> {
        let orig: Vec<char> = text.chars().collect();
        let lower: Option<Vec<char>> = lower_text.map(|t| t.chars().collect());
        let punct: Vec<char> = punctuation.chars().collect();
        py.detach(|| self.scored_for(&orig, lower.as_deref(), &known_words, include_variants, use_word_boundaries, &punct))
            .map_err(PyValueError::new_err)
    }

    /// scored_candidates for many (text, known_words) items on `threads` threads (0: one per CPU).
    #[pyo3(signature = (items, include_variants, use_word_boundaries, punctuation, threads = 0))]
    fn scored_candidates_batch(&self, py: Python<'_>, items: Vec<(String, Vec<KnownWord>, Option<String>)>,
                               include_variants: bool, use_word_boundaries: bool, punctuation: &str,
                               threads: usize) -> PyResult<Vec<Vec<ScoredCandidate>>> {
        let punct: Vec<char> = punctuation.chars().collect();
        let run = || {
            items
                .par_iter()
                .map(|(text, known, lower_text)| {
                    let orig: Vec<char> = text.chars().collect();
                    let lower: Option<Vec<char>> = lower_text.as_ref().map(|t| t.chars().collect());
                    self.scored_for(&orig, lower.as_deref(), known, include_variants, use_word_boundaries, &punct)
                })
                .collect::<Result<Vec<_>, String>>()
        };
        py.detach(|| {
            if threads == 0 {
                run()
            } else {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| e.to_string())?
                    .install(run)
            }
        })
        .map_err(PyValueError::new_err)
    }

    /// Match scores as computed by the Rust backend (for testing against fuzzy_search.tokenization.string).
    #[staticmethod]
    fn scores(phrase: &str, match_string: &str, ngram_size: usize) -> PyResult<(f64, f64, f64)> {
        let p: Vec<char> = phrase.chars().collect();
        let m: Vec<char> = match_string.chars().collect();
        Ok((char_overlap_ratio(&p, &m).map_err(PyValueError::new_err)?,
            ngram_overlap_ratio(&p, &m, ngram_size).map_err(PyValueError::new_err)?,
            levenshtein_ratio(&p, &m)))
    }

    /// Levenshtein ratio as computed by the Rust backend (for testing against Levenshtein.ratio):
    /// the bit-parallel version when `a` has at most 64 characters, else the DP version.
    #[staticmethod]
    #[pyo3(signature = (a, b, bit_parallel = true))]
    fn ratio(a: &str, b: &str, bit_parallel: bool) -> f64 {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        if bit_parallel {
            levenshtein_ratio_with(&a, PatternMasks::new(&a).as_ref(), &b)
        } else {
            levenshtein_ratio(&a, &b)
        }
    }

    /// Number of skipgrams Python's text2skipgrams yields for a text (for testing).
    fn count_text_skipgrams(&self, text: &str) -> usize {
        let chars: Vec<char> = text.chars().collect();
        text_skipgrams(&chars, self.ngram_size, self.skip_size, &self.combos).len()
    }
}

impl SkipgramMatcher {
    /// Candidates in Python's generation order, and the main phrase groups in the order Python's
    /// defaultdict creates them: when a phrase passes the skipgram threshold, even if it yields no
    /// candidates. Later steps break ties by list order, so the group order matters.
    fn candidates_for(&self, chars: &[char], known_words: &[KnownWord],
                      include_variants: bool) -> Result<(Vec<Candidate>, Vec<u32>), String> {
        let mlv = self.max_length_variance;
        let known_at: FxHashMap<i64, usize> =
            known_words.iter().enumerate().map(|(i, kw)| (kw.0, i)).collect();
        let excluded = |ids: &Vec<u32>, id: u32| ids.contains(&id);
        // per phrase: matched skipgrams in scan order, and the distinct skipgram strings
        let mut matched: Vec<Vec<MSkip>> = vec![Vec::new(); self.phrases.len()];
        let mut known: Option<usize> = None;
        for skip in text_skipgrams(chars, self.ngram_size, self.skip_size, &self.combos) {
            if let Some(&i) = known_at.get(&skip.start) {
                known = Some(i);
            }
            if let Some(i) = known {
                if skip.start == known_words[i].1 {
                    known = None;
                }
            }
            if let Some(ids) = self.main_index.get(&skip.key) {
                for &(id, local) in ids {
                    let p = &self.phrases[id as usize];
                    if p.max_start_offset > 0 && p.max_start_end + mlv < skip.start + skip.length {
                        continue;
                    }
                    if p.max_end_offset > 0 && p.max_end_end + mlv < skip.end_offset + skip.length {
                        continue;
                    }
                    if let Some(i) = known {
                        if excluded(&known_words[i].2, id) {
                            continue;
                        }
                    }
                    matched[id as usize].push(MSkip { start: skip.start, length: skip.length, local });
                }
            }
            if include_variants {
                if let Some(ids) = self.variant_index.get(&skip.key) {
                    for &(id, local) in ids {
                        if let Some(i) = known {
                            if excluded(&known_words[i].3, id) {
                                continue;
                            }
                        }
                        matched[id as usize].push(MSkip { start: skip.start, length: skip.length, local });
                    }
                }
            }
        }
        let mut out = Vec::new();
        let mut group_order: Vec<u32> = Vec::new();
        let mut group_seen: FxHashSet<u32> = FxHashSet::default();
        for (id, skips) in matched.iter().enumerate() {
            if skips.is_empty() {
                continue;
            }
            let p = &self.phrases[id];
            let mut seen = vec![false; p.first_offset.len()];
            let mut distinct = 0usize;
            for s in skips {
                if !seen[s.local as usize] {
                    seen[s.local as usize] = true;
                    distinct += 1;
                }
            }
            if (distinct as f64 / p.skipgram_set_size as f64) < self.skipgram_threshold {
                continue;
            }
            let g = self.group[id];
            if group_seen.insert(g) {
                group_order.push(g);
            }
            self.phrase_candidates(id as u32, p, skips, &mut out)?;
        }
        Ok((out, group_order))
    }

    fn filtered_for(&self, orig: &[char], scan: &[char], known_words: &[KnownWord], include_variants: bool,
                    use_word_boundaries: bool, punct: &[char]) -> Result<Vec<Candidate>, String> {
        let (raw, group_order) = self.candidates_for(scan, known_words, include_variants)?;
        // group candidates by main phrase, groups in Python's dict order
        let group_slot: FxHashMap<u32, usize> = group_order.iter().enumerate().map(|(i, &g)| (g, i)).collect();
        let mut groups: Vec<Vec<Candidate>> = vec![Vec::new(); group_order.len()];
        for cand in raw {
            groups[group_slot[&self.group[cand.0 as usize]]].push(cand);
        }
        let mut out = Vec::new();
        for mut list in groups {
            if use_word_boundaries {
                let mut adjusted = Vec::with_capacity(list.len());
                for (id, start, end, overlap) in list {
                    let match_string = slice(orig, start, end);
                    let Some(new_start) = adjust_start(orig, match_string, start) else { continue };
                    let Some(new_end) = adjust_end(&self.phrase_chars[id as usize], match_string, orig, end, punct)?
                        else { continue };
                    if new_end <= new_start {
                        continue;
                    }
                    adjusted.push((id, new_start, new_end, overlap));
                }
                list = adjusted;
            }
            out.extend(self.filter_overlapping(list, orig));
        }
        Ok(out)
    }

    /// Levenshtein.ratio(phrase string of `id`, text)
    fn ratio_to(&self, id: u32, text: &[char]) -> f64 {
        levenshtein_ratio_with(&self.phrase_chars[id as usize], self.phrase_masks[id as usize].as_ref(), text)
    }

    fn scored_for(&self, orig: &[char], lower: Option<&[char]>, known_words: &[KnownWord], include_variants: bool,
                  use_word_boundaries: bool, punct: &[char]) -> Result<Vec<ScoredCandidate>, String> {
        let scan = lower.unwrap_or(orig);
        let cands = self.filtered_for(orig, scan, known_words, include_variants, use_word_boundaries, punct)?;
        self.score(cands, lower.unwrap_or(orig), lower.is_some())
    }

    /// PhraseMatch.add_scores: scores between the phrase string and the match string, both
    /// lowercased when `lowercase` (then `text` is the lowercased text).
    fn score(&self, cands: Vec<Candidate>, text: &[char], lowercase: bool) -> Result<Vec<ScoredCandidate>, String> {
        cands
            .into_iter()
            .map(|(id, start, end, overlap)| {
                let i = id as usize;
                let (p, masks) = if lowercase {
                    (&self.phrase_lower_chars[i], self.phrase_lower_masks[i].as_ref())
                } else {
                    (&self.phrase_chars[i], self.phrase_masks[i].as_ref())
                };
                let m = slice(text, start, end);
                Ok((id, start, end, overlap, char_overlap_ratio(p, m)?,
                    ngram_overlap_ratio(p, m, self.ngram_size)?, levenshtein_ratio_with(p, masks, m)))
            })
            .collect()
    }

    /// Mirror of skip_match.filter_overlapping_phrase_candidates for the candidates of one main phrase.
    fn filter_overlapping(&self, mut list: Vec<Candidate>, orig: &[char]) -> Vec<Candidate> {
        if list.len() < 2 {
            return list;
        }
        let tie_key = |c: &Candidate| (!self.is_main[c.0 as usize], &self.phrase_chars[c.0 as usize]);
        list.sort_by(|a, b| (a.1, tie_key(a)).cmp(&(b.1, tie_key(b))));
        let score = |c: &Candidate| self.ratio_to(c.0, slice(orig, c.1, c.2));
        let len = |c: &Candidate| slice(orig, c.1, c.2).len();
        let mut filtered: Vec<usize> = Vec::new();
        let mut prev = 0usize;
        let mut prev_score = score(&list[0]);
        for ci in 1..list.len() {
            let curr = &list[ci];
            let p = &list[prev];
            if curr.2 > p.1 {
                if curr.1 < p.2 {
                    let curr_score = score(curr);
                    if curr_score > prev_score
                        || (curr_score == prev_score && len(curr) > len(p))
                        || (curr_score == prev_score && len(curr) == len(p) && tie_key(curr) < tie_key(p))
                    {
                        prev = ci;
                        prev_score = curr_score;
                    }
                } else {
                    filtered.push(prev);
                    prev = ci;
                    prev_score = score(&list[ci]);
                }
            }
        }
        if filtered.last() != Some(&prev) {
            filtered.push(prev);
        }
        filtered.into_iter().map(|i| list[i]).collect()
    }

    /// Mirror of skip_match.get_skipmatch_phrase_candidates.
    fn phrase_candidates(&self, id: u32, p: &PhraseData, skips: &[MSkip],
                         out: &mut Vec<Candidate>) -> Result<(), String> {
        let mlv = self.max_length_variance;
        let threshold = self.skipgram_threshold;
        let gap = (self.ngram_size + self.skip_size + 1) as i64;
        let first = out.len();
        let mut partial = Partial::new(p, mlv);
        for (ci, skip) in skips.iter().enumerate() {
            partial.add(*skip);
            if partial.is_match(threshold, mlv) {
                check_match_string(partial.start.unwrap(), partial.end)?;
                if out.len() == first || !partial.same_as(out.last().unwrap()) {
                    out.push(partial.to_candidate(id)?);
                }
                if partial.shift_start_skip() {
                    out.push(partial.to_candidate(id)?);
                }
            }
            if let Some(next) = skips.get(ci + 1) {
                // Python: `if next_start_offset and ...`, so a next offset of 0 never splits
                if next.start != 0 && next.start - skip.start > gap {
                    partial.reset();
                }
            }
        }
        if partial.is_match(threshold, mlv) {
            if out.len() == first || !partial.same_as(out.last().unwrap()) {
                out.push(partial.to_candidate(id)?);
            }
            if partial.shift_start_skip() {
                out.push(partial.to_candidate(id)?);
            }
        }
        Ok(())
    }
}

#[pymodule]
fn fuzzy_search_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SkipgramMatcher>()?;
    m.add("MAX_NGRAM_SIZE", MAX_NGRAM_SIZE)?;
    Ok(())
}
