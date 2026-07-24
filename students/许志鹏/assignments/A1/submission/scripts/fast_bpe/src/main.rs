use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Pair(u32, u32);

#[derive(Clone, Debug, Eq, PartialEq)]
struct PairCount {
    count: u64,
    pair: Pair,
    left: Vec<u8>,
    right: Vec<u8>,
}

#[derive(Debug)]
struct MergeUpdate {
    old_id: usize,
    old_word: Vec<u32>,
    new_word: Vec<u32>,
    count: u64,
    old_pairs: Vec<Pair>,
    new_pairs: Vec<Pair>,
}

const PARALLEL_AFFECTED_THRESHOLD: usize = 2048;

impl Ord for PairCount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.left.cmp(&other.left))
            .then_with(|| self.right.cmp(&other.right))
    }
}

impl PartialOrd for PairCount {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_space(ch: char) -> bool {
    ch.is_whitespace()
}

fn next_char(text: &str, idx: usize) -> Option<(char, usize)> {
    text[idx..].chars().next().map(|ch| (ch, ch.len_utf8()))
}

fn starts_with_contraction(text: &str, idx: usize) -> Option<usize> {
    let rest = &text[idx..];
    for suffix in ["'ll", "'ve", "'re", "'s", "'d", "'m", "'t"] {
        if rest.starts_with(suffix) {
            return Some(suffix.len());
        }
    }
    None
}

fn consume_run<F>(text: &str, mut idx: usize, pred: F) -> usize
where
    F: Fn(char) -> bool,
{
    while idx < text.len() {
        let Some((ch, width)) = next_char(text, idx) else {
            break;
        };
        if !pred(ch) {
            break;
        }
        idx += width;
    }
    idx
}

fn consume_whitespace_like_gpt2(text: &str, start: usize) -> (usize, usize) {
    let mut idx = start;
    let mut last_start = start;
    let mut count = 0usize;
    while idx < text.len() {
        let Some((ch, width)) = next_char(text, idx) else {
            break;
        };
        if !is_space(ch) {
            break;
        }
        last_start = idx;
        idx += width;
        count += 1;
    }
    if idx < text.len() && count > 1 {
        (last_start, last_start)
    } else {
        (idx, idx)
    }
}

fn pretokenize_segment(text: &str, counts: &mut HashMap<Vec<u8>, u64>) {
    let mut idx = 0;
    while idx < text.len() {
        if let Some(width) = starts_with_contraction(text, idx) {
            add_pretoken(&text.as_bytes()[idx..idx + width], counts);
            idx += width;
            continue;
        }

        let start = idx;
        let Some((ch, width)) = next_char(text, idx) else {
            break;
        };
        if ch == ' ' {
            if let Some((next, next_width)) = next_char(text, idx + width) {
                if is_letter(next) {
                    idx = consume_run(text, idx + width + next_width, is_letter);
                    add_pretoken(&text.as_bytes()[start..idx], counts);
                    continue;
                }
                if is_number(next) {
                    idx = consume_run(text, idx + width + next_width, is_number);
                    add_pretoken(&text.as_bytes()[start..idx], counts);
                    continue;
                }
                if !is_space(next) && !is_letter(next) && !is_number(next) {
                    idx = consume_run(text, idx + width + next_width, |c| {
                        !is_space(c) && !is_letter(c) && !is_number(c)
                    });
                    add_pretoken(&text.as_bytes()[start..idx], counts);
                    continue;
                }
            }
        }

        if is_letter(ch) {
            idx = consume_run(text, idx + width, is_letter);
        } else if is_number(ch) {
            idx = consume_run(text, idx + width, is_number);
        } else if !is_space(ch) {
            idx = consume_run(text, idx + width, |c| {
                !is_space(c) && !is_letter(c) && !is_number(c)
            });
        } else {
            let (match_end, next_idx) = consume_whitespace_like_gpt2(text, idx);
            add_pretoken(&text.as_bytes()[start..match_end], counts);
            idx = next_idx;
            continue;
        }
        add_pretoken(&text.as_bytes()[start..idx], counts);
    }
}

fn add_pretoken(bytes: &[u8], counts: &mut HashMap<Vec<u8>, u64>) {
    if !bytes.is_empty() {
        *counts.entry(bytes.to_vec()).or_insert(0) += 1;
    }
}

fn merge_pretoken_counts(
    mut left: HashMap<Vec<u8>, u64>,
    right: HashMap<Vec<u8>, u64>,
) -> HashMap<Vec<u8>, u64> {
    for (word, count) in right {
        *left.entry(word).or_insert(0) += count;
    }
    left
}

fn split_specials<'a>(text: &'a str, special_tokens: &[String]) -> Vec<&'a str> {
    if special_tokens.is_empty() {
        return vec![text];
    }
    if special_tokens.len() == 1 {
        return text
            .split(special_tokens[0].as_str())
            .filter(|segment| !segment.is_empty())
            .collect();
    }
    let mut segments = Vec::new();
    let mut idx = 0;
    while idx < text.len() {
        let mut best: Option<(&str, usize)> = None;
        for token in special_tokens {
            if text[idx..].starts_with(token) {
                let len = token.len();
                if best.map_or(true, |(_, old_len)| len > old_len) {
                    best = Some((token.as_str(), len));
                }
            }
        }
        if let Some((_, len)) = best {
            idx += len;
            continue;
        }
        let start = idx;
        while idx < text.len() {
            let mut found = false;
            for token in special_tokens {
                if text[idx..].starts_with(token) {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
            let Some((_, width)) = next_char(text, idx) else {
                break;
            };
            idx += width;
        }
        if start < idx {
            segments.push(&text[start..idx]);
        }
    }
    segments
}

fn pairs_in_word(word: &[u32]) -> Vec<Pair> {
    word.windows(2).map(|pair| Pair(pair[0], pair[1])).collect()
}

fn build_indexes(
    words: &[Vec<u32>],
    word_counts: &[u64],
) -> (HashMap<Pair, u64>, HashMap<Pair, HashSet<usize>>) {
    (0..words.len())
        .into_par_iter()
        .fold(
            || {
                (
                    HashMap::<Pair, u64>::default(),
                    HashMap::<Pair, HashSet<usize>>::default(),
                )
            },
            |(mut pair_counts, mut pair_to_words), word_id| {
                let count = word_counts[word_id];
                for pair in words[word_id].windows(2) {
                    let p = Pair(pair[0], pair[1]);
                    *pair_counts.entry(p).or_insert(0) += count;
                    pair_to_words.entry(p).or_default().insert(word_id);
                }
                (pair_counts, pair_to_words)
            },
        )
        .reduce(
            || {
                (
                    HashMap::<Pair, u64>::default(),
                    HashMap::<Pair, HashSet<usize>>::default(),
                )
            },
            |(mut left_counts, mut left_words), (right_counts, right_words)| {
                for (pair, count) in right_counts {
                    *left_counts.entry(pair).or_insert(0) += count;
                }
                for (pair, words) in right_words {
                    left_words.entry(pair).or_default().extend(words);
                }
                (left_counts, left_words)
            },
        )
}

fn build_pair_heap(pair_counts: &HashMap<Pair, u64>, vocab: &[Vec<u8>]) -> BinaryHeap<PairCount> {
    pair_counts
        .par_iter()
        .map(|(pair, count)| make_pair_count(*pair, *count, vocab))
        .collect()
}

fn make_pair_count(pair: Pair, count: u64, vocab: &[Vec<u8>]) -> PairCount {
    PairCount {
        count,
        pair,
        left: vocab[pair.0 as usize].clone(),
        right: vocab[pair.1 as usize].clone(),
    }
}

fn push_pair_count(heap: &mut BinaryHeap<PairCount>, pair: Pair, count: u64, vocab: &[Vec<u8>]) {
    if count > 0 {
        heap.push(make_pair_count(pair, count, vocab));
    }
}

fn dec_pair_count(
    pair_counts: &mut HashMap<Pair, u64>,
    heap: &mut BinaryHeap<PairCount>,
    pair: Pair,
    amount: u64,
    vocab: &[Vec<u8>],
) {
    let mut next_count = None;
    let mut should_remove = false;
    if let Some(value) = pair_counts.get_mut(&pair) {
        if *value > amount {
            *value -= amount;
            next_count = Some(*value);
        } else {
            should_remove = true;
        }
    }
    if should_remove {
        pair_counts.remove(&pair);
    } else if let Some(count) = next_count {
        push_pair_count(heap, pair, count, vocab);
    }
}

fn inc_pair_count(
    pair_counts: &mut HashMap<Pair, u64>,
    heap: &mut BinaryHeap<PairCount>,
    pair: Pair,
    amount: u64,
    vocab: &[Vec<u8>],
) {
    let count = pair_counts.entry(pair).or_insert(0);
    *count += amount;
    push_pair_count(heap, pair, *count, vocab);
}

fn best_pair(heap: &mut BinaryHeap<PairCount>, pair_counts: &HashMap<Pair, u64>) -> Option<Pair> {
    while let Some(entry) = heap.pop() {
        if pair_counts
            .get(&entry.pair)
            .is_some_and(|count| *count == entry.count)
        {
            return Some(entry.pair);
        }
    }
    None
}

fn merge_word(word: &[u32], pair: Pair, new_token: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
            merged.push(new_token);
            i += 2;
        } else {
            merged.push(word[i]);
            i += 1;
        }
    }
    merged
}

fn merge_update(
    old_id: usize,
    old_word: Vec<u32>,
    count: u64,
    pair: Pair,
    new_token: u32,
) -> MergeUpdate {
    let old_pairs = pairs_in_word(&old_word);
    let new_word = merge_word(&old_word, pair, new_token);
    let new_pairs = pairs_in_word(&new_word);
    MergeUpdate {
        old_id,
        old_word,
        new_word,
        count,
        old_pairs,
        new_pairs,
    }
}

fn hex(bytes: &[u8]) -> String {
    const CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(CHARS[(b >> 4) as usize] as char);
        out.push(CHARS[(b & 0x0f) as usize] as char);
    }
    out
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_json(
    path: &str,
    vocab: &[Vec<u8>],
    merges: &[(Vec<u8>, Vec<u8>)],
    special_tokens: &[String],
) -> io::Result<()> {
    let mut out = String::new();
    out.push_str("{\"vocab\":[");
    for (idx, token) in vocab.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&format!("[{},\"{}\"]", idx, hex(token)));
    }
    out.push_str("],\"merges\":[");
    for (idx, pair) in merges.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&format!("[\"{}\",\"{}\"]", hex(&pair.0), hex(&pair.1)));
    }
    out.push_str("],\"special_tokens\":[");
    for (idx, token) in special_tokens.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(token));
        out.push('"');
    }
    out.push_str("]}");
    fs::write(path, out)
}

fn parse_args() -> Result<
    (
        String,
        String,
        usize,
        Vec<String>,
        usize,
        usize,
        Option<String>,
    ),
    String,
> {
    let mut input = None;
    let mut output = None;
    let mut vocab_size = None;
    let mut special_tokens = Vec::new();
    let mut progress_interval = 500usize;
    let mut num_workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut dump_pretokens = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next(),
            "--output-json" => output = args.next(),
            "--vocab-size" => {
                vocab_size = Some(
                    args.next()
                        .ok_or("missing --vocab-size value")?
                        .parse()
                        .map_err(|_| "bad vocab size")?,
                )
            }
            "--special-token" => {
                special_tokens.push(args.next().ok_or("missing --special-token value")?)
            }
            "--progress-interval" => {
                progress_interval = args
                    .next()
                    .ok_or("missing --progress-interval value")?
                    .parse()
                    .map_err(|_| "bad progress interval")?;
            }
            "--num-workers" => {
                num_workers = args
                    .next()
                    .ok_or("missing --num-workers value")?
                    .parse()
                    .map_err(|_| "bad num workers")?;
                if num_workers == 0 {
                    return Err("--num-workers must be positive".to_string());
                }
            }
            "--dump-pretokens" => dump_pretokens = args.next(),
            _ => return Err(format!("unknown argument: {}", arg)),
        }
    }
    Ok((
        input.ok_or("missing --input")?,
        output.ok_or("missing --output-json")?,
        vocab_size.ok_or("missing --vocab-size")?,
        special_tokens,
        progress_interval,
        num_workers,
        dump_pretokens,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, output, vocab_size, special_tokens, progress_interval, num_workers, dump_pretokens) =
        parse_args().map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    let start = Instant::now();
    let text = fs::read_to_string(&input)?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let segments = split_specials(&text, &special_tokens);
    let threads = rayon::current_num_threads();
    let chunk_size = (segments.len() / (threads * 16)).max(1);
    let pretoken_counts: HashMap<Vec<u8>, u64> = segments
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut counts = HashMap::default();
            for segment in chunk {
                pretokenize_segment(segment, &mut counts);
            }
            counts
        })
        .reduce(HashMap::default, merge_pretoken_counts);
    eprintln!(
        "pretokenized unique_words={} segments={} chunk_size={} workers={} elapsed={:.1}s",
        pretoken_counts.len(),
        segments.len(),
        chunk_size,
        threads,
        start.elapsed().as_secs_f64()
    );

    if let Some(path) = dump_pretokens {
        let mut lines: Vec<String> = pretoken_counts
            .iter()
            .map(|(bytes, count)| format!("{}\t{}", hex(bytes), count))
            .collect();
        lines.sort();
        fs::write(path, lines.join("\n"))?;
    }

    let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
    for token in &special_tokens {
        let bytes = token.as_bytes().to_vec();
        if !vocab.iter().any(|item| item == &bytes) {
            vocab.push(bytes);
        }
    }

    let convert_start = Instant::now();
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(pretoken_counts.len());
    let mut word_counts: Vec<u64> = Vec::with_capacity(pretoken_counts.len());
    let mut active: Vec<bool> = Vec::with_capacity(pretoken_counts.len());
    let mut word_to_id: HashMap<Vec<u32>, usize> =
        HashMap::with_capacity_and_hasher(pretoken_counts.len(), Default::default());
    for (bytes, count) in pretoken_counts {
        let word: Vec<u32> = bytes.into_iter().map(u32::from).collect();
        let id = words.len();
        words.push(word.clone());
        word_counts.push(count);
        active.push(true);
        word_to_id.insert(word, id);
    }
    eprintln!(
        "converted id_words={} elapsed={:.1}s total_elapsed={:.1}s",
        words.len(),
        convert_start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );

    let index_start = Instant::now();
    let (mut pair_counts, mut pair_to_words) = build_indexes(&words, &word_counts);
    let mut pair_heap = build_pair_heap(&pair_counts, &vocab);
    eprintln!(
        "indexed pair_counts={} heap_entries={} workers={} elapsed={:.1}s total_elapsed={:.1}s",
        pair_counts.len(),
        pair_heap.len(),
        rayon::current_num_threads(),
        index_start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    while vocab.len() < vocab_size {
        let merge_start = Instant::now();
        let stale_before = pair_heap.len();
        let Some(pair) = best_pair(&mut pair_heap, &pair_counts) else {
            break;
        };
        let stale_popped = stale_before.saturating_sub(pair_heap.len() + 1);
        let left_bytes = vocab[pair.0 as usize].clone();
        let right_bytes = vocab[pair.1 as usize].clone();
        let mut merged_token = left_bytes.clone();
        merged_token.extend_from_slice(&right_bytes);
        let new_token = vocab.len() as u32;
        vocab.push(merged_token);
        merges.push((left_bytes, right_bytes));

        let affected: Vec<usize> = pair_to_words
            .remove(&pair)
            .unwrap_or_default()
            .into_iter()
            .filter(|word_id| active[*word_id])
            .collect();
        let affected_with_counts: Vec<(usize, u64, Vec<u32>)> = affected
            .into_iter()
            .map(|word_id| (word_id, word_counts[word_id], words[word_id].clone()))
            .collect();

        let heap_before_update = pair_heap.len();
        let affected_count = affected_with_counts.len();
        let mut old_pair_deltas: HashMap<Pair, u64> = HashMap::default();
        let mut new_pair_deltas: HashMap<Pair, u64> = HashMap::default();
        let updates: Vec<MergeUpdate> = if affected_count >= PARALLEL_AFFECTED_THRESHOLD {
            affected_with_counts
                .into_par_iter()
                .map(|(word_id, count, old_word)| {
                    merge_update(word_id, old_word, count, pair, new_token)
                })
                .collect()
        } else {
            affected_with_counts
                .into_iter()
                .map(|(word_id, count, old_word)| {
                    merge_update(word_id, old_word, count, pair, new_token)
                })
                .collect()
        };

        for update in &updates {
            active[update.old_id] = false;
            word_counts[update.old_id] = 0;
            word_to_id.remove(&update.old_word);
            for old_pair in &update.old_pairs {
                *old_pair_deltas.entry(*old_pair).or_insert(0) += update.count;
                if *old_pair != pair {
                    if let Some(ids) = pair_to_words.get_mut(old_pair) {
                        ids.remove(&update.old_id);
                        if ids.is_empty() {
                            pair_to_words.remove(old_pair);
                        }
                    }
                }
            }
        }

        for update in updates {
            let target_id = if let Some(id) = word_to_id.get(&update.new_word) {
                *id
            } else {
                let id = words.len();
                words.push(update.new_word.clone());
                word_counts.push(0);
                active.push(true);
                word_to_id.insert(update.new_word.clone(), id);
                id
            };
            word_counts[target_id] += update.count;
            for new_pair in update.new_pairs {
                *new_pair_deltas.entry(new_pair).or_insert(0) += update.count;
                pair_to_words.entry(new_pair).or_default().insert(target_id);
            }
        }

        let changed_pairs = old_pair_deltas.len() + new_pair_deltas.len();
        for (old_pair, delta) in old_pair_deltas {
            dec_pair_count(&mut pair_counts, &mut pair_heap, old_pair, delta, &vocab);
        }
        for (new_pair, delta) in new_pair_deltas {
            inc_pair_count(&mut pair_counts, &mut pair_heap, new_pair, delta, &vocab);
        }
        let mut compacted_heap = false;
        if pair_heap.len() > pair_counts.len().saturating_mul(8).max(1_000_000) {
            pair_heap = build_pair_heap(&pair_counts, &vocab);
            compacted_heap = true;
        }

        if progress_interval > 0 && merges.len() % progress_interval == 0 {
            let active_words = active.iter().filter(|flag| **flag).count();
            eprintln!(
                "merge {}/{} pair_counts={} heap_entries={} compacted_heap={} stale_popped={} heap_delta={} changed_pairs={} affected_words={} active_words={} merge_elapsed={:.2}s workers={} elapsed={:.1}s",
                merges.len(),
                vocab_size.saturating_sub(256 + special_tokens.len()),
                pair_counts.len(),
                pair_heap.len(),
                compacted_heap,
                stale_popped,
                pair_heap.len() as i128 - heap_before_update as i128,
                changed_pairs,
                affected_count,
                active_words,
                merge_start.elapsed().as_secs_f64(),
                rayon::current_num_threads(),
                start.elapsed().as_secs_f64()
            );
            io::stderr().flush().ok();
        }
    }
    write_json(&output, &vocab, &merges, &special_tokens)?;
    eprintln!(
        "done vocab={} merges={} elapsed={:.1}s",
        vocab.len(),
        merges.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
