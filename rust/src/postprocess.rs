//! Mirror of the candidate post-processing in FuzzyPhraseSearcher.filter_phrase_candidates:
//! word-boundary adjustment (match_offsets.adjust_match_offsets) and per-phrase overlap
//! filtering (skip_match.filter_overlapping_phrase_candidates).

/// Python slice semantics for non-negative bounds: s[a:b] clamped to the string.
pub fn slice(s: &[char], a: i64, b: i64) -> &[char] {
    let n = s.len() as i64;
    let a = a.clamp(0, n);
    let b = b.clamp(0, n);
    if b <= a { &s[0..0] } else { &s[a as usize..b as usize] }
}

fn tail(s: &[char], k: usize) -> &[char] {
    &s[s.len().saturating_sub(k)..]
}

fn is_affix_1(c: char) -> bool {
    matches!(c, ' ' | '.' | ',' | '!' | '?')
}

fn affix_1(s: &[char]) -> bool {
    s.len() == 1 && is_affix_1(s[0])
}

fn affix_2(s: &[char]) -> bool {
    if s.len() != 2 {
        return false;
    }
    matches!(
        (s[0], s[1]),
        ('.', ' ') | (',', ' ') | ('!', ' ') | ('?', ' ')
            | (' ', '(') | (')', ' ') | (')', '.') | (')', '!') | (')', ',') | (')', '?')
            | (' ', '[') | (']', ' ') | (']', '.') | (']', '!') | (']', ',') | (']', '?')
    )
}

/// Length of string.get_non_word_prefix(s).
fn non_word_prefix_len(s: &[char]) -> i64 {
    if affix_2(slice(s, 0, 2)) {
        2
    } else if affix_1(slice(s, 0, 1)) {
        1
    } else {
        0
    }
}

#[allow(clippy::if_same_then_else)] // kept literal to match match_offsets.adjust_match_start_offset
pub fn adjust_start(text: &[char], match_string: &[char], offset: i64) -> Option<i64> {
    let prefix = non_word_prefix_len(match_string);
    if prefix > 0 {
        return Some(offset + prefix);
    }
    if offset == 0 {
        return Some(offset);
    }
    // text['text'][0] raises IndexError on an empty text in Python; offset > 0 implies a non-empty text
    let first_not_affix = !text.is_empty() && !is_affix_1(text[0]);
    if offset == 1 && first_not_affix {
        return Some(offset - 1);
    }
    if affix_1(slice(text, offset - 1, offset)) {
        Some(offset)
    } else if offset > 1 && affix_1(slice(text, offset - 2, offset - 1)) {
        Some(offset - 1)
    } else if offset == 2 && first_not_affix {
        Some(offset - 2)
    } else if offset > 2 && affix_1(slice(text, offset - 3, offset - 2)) {
        Some(offset - 2)
    } else {
        None
    }
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

fn map_string(s: &[char], punctuation: &[char], whitespace_only: bool) -> String {
    s.iter()
        .map(|&c| if is_ws(c) || (!whitespace_only && punctuation.contains(&c)) { 's' } else { 'w' })
        .collect()
}

pub fn adjust_end(phrase: &[char], candidate: &[char], text: &[char], end: i64,
                  punctuation: &[char]) -> Result<Option<i64>, String> {
    let last = *phrase.last().ok_or("empty phrase string")?;
    let whitespace_only = if punctuation.contains(&last) {
        true
    } else if is_ws(last) {
        let before = *phrase.get(phrase.len().wrapping_sub(2)).ok_or("string index out of range")?;
        punctuation.contains(&before)
    } else {
        false
    };
    let phrase_end = map_string(tail(phrase, 3), punctuation, whitespace_only);
    let match_end = map_string(tail(candidate, 3), punctuation, whitespace_only);
    let text_suffix = map_string(slice(text, end, end + 3), punctuation, whitespace_only);
    calculate_end_shift(&phrase_end, &match_end, &text_suffix, end)
}

/// Literal port of match_offsets.calculate_end_shift.
/// Kept literal, including branches with equal results, so it can be checked against the Python.
#[allow(clippy::if_same_then_else)]
fn calculate_end_shift(phrase_end: &str, match_end: &str, text_suffix: &str, end: i64) -> Result<Option<i64>, String> {
    let ts = text_suffix;
    let starts = |p: &str| ts.starts_with(p);
    if phrase_end == match_end && (ts.is_empty() || starts("s")) {
        return Ok(Some(end));
    }
    if phrase_end.ends_with('s') && match_end.ends_with('s') {
        return Ok(Some(end));
    }
    if match_end == "wss" {
        return Ok(Some(end - 2));
    }
    if phrase_end == "www" {
        if match_end == "www" {
            if ts == "w" || starts("ws") {
                return Ok(Some(end + 1));
            } else if ts == "ww" || starts("wws") {
                return Ok(Some(end + 2));
            } else if starts("www") {
                return Ok(None);
            }
        }
        if match_end == "wws" {
            return Ok(Some(end - 1));
        }
        if match_end == "wsw" {
            if ts.is_empty() || starts("s") {
                return Ok(Some(end));
            }
            if starts("w") {
                return Ok(Some(end - 2));
            }
        }
        if match_end == "sww" {
            if ts.is_empty() || starts("s") {
                return Ok(Some(end));
            } else if starts("w") {
                return Ok(None);
            }
        }
        if match_end == "sws" {
            return Ok(Some(end - 1));
        }
        if match_end == "ssw" {
            return Ok(None);
        } else {
            return Ok(None);
        }
    }
    if phrase_end == "wws" {
        return Ok(if match_end == "www" {
            if ts.is_empty() {
                Some(end)
            } else if starts("s") {
                Some(end + 1)
            } else if ts == "w" {
                None
            } else if starts("ws") {
                Some(end + 2)
            } else {
                None
            }
        } else if match_end.starts_with("ws") {
            Some(end - 1)
        } else if match_end.starts_with('s') {
            Some(end - 2)
        } else {
            None
        });
    }
    if phrase_end == "sww" {
        return Ok(match match_end {
            "sww" => if ts == "w" || starts("ws") { Some(end + 1) } else { None },
            "sws" => Some(end - 1),
            "www" => if ts.is_empty() || starts("s") { Some(end) } else { None },
            "wsw" => {
                if ts.is_empty() || starts("s") {
                    Some(end)
                } else if ts == "ws" || starts("ws") {
                    Some(end + 1)
                } else if ts == "ww" || starts("wws") {
                    Some(end + 2)
                } else {
                    None
                }
            }
            "ssw" => {
                if ts.is_empty() || starts("s") {
                    Some(end)
                } else if ts == "w" || starts("ws") {
                    Some(end + 1)
                } else if ts == "ww" || starts("wws") {
                    Some(end + 1)
                } else {
                    None
                }
            }
            _ => None,
        });
    }
    if phrase_end == "sws" {
        return Ok(match match_end {
            "www" => if ts == "sw" || ts == "sws" { Some(end + 2) } else { None },
            "sww" => Some(end - 2),
            "wsw" => {
                if ts.is_empty() {
                    Some(end)
                } else if starts("s") {
                    Some(end + 1)
                } else {
                    Some(end - 1)
                }
            }
            _ => None,
        });
    }
    if phrase_end == "wsw" {
        if match_end == "wsw" {
            return Ok(if ts == "w" || starts("ws") { Some(end + 1) } else { None });
        }
        if match_end == "www" {
            return Ok(if ts.is_empty() || ts == "s" {
                Some(end)
            } else if starts("w") {
                None
            } else if ts == "sw" || ts == "sws" {
                Some(end + 2)
            } else {
                None
            });
        }
        if match_end == "sww" {
            return Ok(if ts.is_empty() {
                Some(end + 1)
            } else if starts("s") {
                Some(end)
            } else {
                None
            });
        }
        if match_end == "ssw" {
            return Ok(if ts.is_empty() || starts("s") { Some(end) } else { None });
        } else {
            return Ok(None);
        }
    }
    if phrase_end.chars().count() < 3 {
        return Ok(if phrase_end == match_end { Some(end) } else { None });
    }
    Err(format!("combination not captured: phrase_end {phrase_end}, match_end {match_end}, text_suffix {text_suffix}"))
}

/// Levenshtein.ratio (Indel normalized similarity): 1 - (len1 + len2 - 2 * LCS) / (len1 + len2).
pub fn levenshtein_ratio(a: &[char], b: &[char]) -> f64 {
    let lensum = a.len() + b.len();
    if lensum == 0 {
        return 1.0;
    }
    let dist = lensum - 2 * lcs_len(a, b);
    1.0 - dist as f64 / lensum as f64
}

fn lcs_len(a: &[char], b: &[char]) -> usize {
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    let mut row = vec![0usize; b.len() + 1];
    for &ca in a {
        let mut diag = 0;
        for j in 0..b.len() {
            let up = row[j + 1];
            row[j + 1] = if ca == b[j] { diag + 1 } else { up.max(row[j]) };
            diag = up;
        }
    }
    row[b.len()]
}

/// string.score_char_overlap_ratio(phrase, match): multiset character intersection / len(phrase).
pub fn char_overlap_ratio(phrase: &[char], m: &[char]) -> Result<f64, String> {
    if phrase.is_empty() {
        return Err("division by zero".to_string());
    }
    let mut counts: rustc_hash::FxHashMap<char, i64> = rustc_hash::FxHashMap::default();
    for &c in phrase {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut overlap = 0i64;
    for c in m {
        if let Some(n) = counts.get_mut(c) {
            if *n > 0 {
                *n -= 1;
                overlap += 1;
            }
        }
    }
    Ok(overlap as f64 / phrase.len() as f64)
}

/// string.score_ngram_overlap_ratio(phrase, match, n): multiset intersection of padded ngrams
/// divided by the number of ngrams of the phrase.
pub fn ngram_overlap_ratio(phrase: &[char], m: &[char], n: usize) -> Result<f64, String> {
    let padded = |t: &[char]| -> Option<Vec<char>> {
        if n > t.len() {
            return None;
        }
        let mut v = Vec::with_capacity(t.len() + 2);
        v.push('#');
        v.extend_from_slice(t);
        v.push('#');
        Some(v)
    };
    let Some(p) = padded(phrase) else { return Err("division by zero".to_string()) };
    let p_ngrams: Vec<&[char]> = p.windows(n).collect();
    let mut counts: rustc_hash::FxHashMap<&[char], i64> = rustc_hash::FxHashMap::default();
    let overlap = match padded(m) {
        None => 0,
        Some(mp) => {
            for g in mp.windows(n) {
                *counts.entry(g).or_insert(0) += 1;
            }
            let mut overlap = 0i64;
            for g in &p_ngrams {
                if let Some(c) = counts.get_mut(g) {
                    if *c > 0 {
                        *c -= 1;
                        overlap += 1;
                    }
                }
            }
            overlap
        }
    };
    Ok(overlap as f64 / p_ngrams.len() as f64)
}

/// Precomputed bit masks of a pattern for bit-parallel LCS: bit i of mask[c] is set when
/// pattern[i] == c. Only for patterns of at most 64 characters.
pub struct PatternMasks {
    len: usize,
    masks: rustc_hash::FxHashMap<char, u64>,
}

impl PatternMasks {
    pub fn new(pattern: &[char]) -> Option<Self> {
        if pattern.len() > 64 {
            return None;
        }
        let mut masks: rustc_hash::FxHashMap<char, u64> = rustc_hash::FxHashMap::default();
        for (i, &c) in pattern.iter().enumerate() {
            *masks.entry(c).or_insert(0) |= 1u64 << i;
        }
        Some(PatternMasks { len: pattern.len(), masks })
    }

    /// Length of the longest common subsequence with `text` (Hyyrö's bit-vector algorithm).
    pub fn lcs_len(&self, text: &[char]) -> usize {
        if self.len == 0 {
            return 0;
        }
        let full = if self.len == 64 { u64::MAX } else { (1u64 << self.len) - 1 };
        let mut v = u64::MAX;
        for c in text {
            if let Some(&m) = self.masks.get(c) {
                let u = v & m;
                v = v.wrapping_add(u) | v.wrapping_sub(u);
            }
        }
        (!v & full).count_ones() as usize
    }
}

/// levenshtein_ratio(pattern, text) using the pattern's precomputed masks when available.
pub fn levenshtein_ratio_with(pattern: &[char], masks: Option<&PatternMasks>, text: &[char]) -> f64 {
    let Some(masks) = masks else { return levenshtein_ratio(pattern, text) };
    let lensum = pattern.len() + text.len();
    if lensum == 0 {
        return 1.0;
    }
    let dist = lensum - 2 * masks.lcs_len(text);
    1.0 - dist as f64 / lensum as f64
}
