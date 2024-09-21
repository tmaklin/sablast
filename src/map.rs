// sablast: Spectral Burrows-Wheeler transform accelerated local alignment search
//
// Copyright 2024 Tommi Mäklin [tommi@maklin.fi].

// Copyrights in this project are retained by contributors. No copyright assignment
// is required to contribute to this project.

// Except as otherwise noted (below and/or in individual files), this
// project is licensed under the Apache License, Version 2.0
// <LICENSE-APACHE> or <http://www.apache.org/licenses/LICENSE-2.0> or
// the MIT license, <LICENSE-MIT> or <http://opensource.org/licenses/MIT>,
// at your option.
//
// Parameters for SBWT construction
#[derive(Clone)]
pub struct TranslateParams {
    pub k: usize,
    pub threshold: usize,
}

fn log_rm_max_cdf(
    t: usize,
    alphabet_size: usize,
    n_kmers: usize,
) -> f64 {
    n_kmers as f64 * (- ((1.0_f64.ln() - (alphabet_size as f64).ln()).exp()).powi(t as i32 + 1)).ln_1p()
}

pub fn random_match_threshold(
    k: usize,
    n_kmers: usize,
    alphabet_size: usize,
    max_error_prob: f64,
) -> usize {
    for i in 1..k {
	if log_rm_max_cdf(i, alphabet_size, n_kmers) > (-max_error_prob).ln_1p() {
	    return i;
	}
    }
    return k;
}

fn ms_to_run(
    curr: usize,
    next: usize,
    next_run: i64,
    threshold: usize,
    k: usize,
) -> i64 {
    let run: i64 = if curr == k && next == k {
	k as i64
    } else if curr == k && next_run == 1 {
	k as i64
    } else if curr == k && next_run < 0 {
	k as i64
    } else if curr < threshold {
	next_run - 1
    } else if curr > threshold && next_run <= 0 {
	curr as i64
    } else if curr > threshold && next_run == 1 {
	curr as i64
    } else if curr > threshold && next_run < curr as i64 {
	curr as i64
    } else {
	next_run - 1
    };

    return run;
}

fn run_to_aln(
    runs: &Vec<i64>,
    curr_ms: usize,
    threshold: usize,
    k: usize,
    res: &mut Vec<char>,
    pos: &mut usize,
) {
    let prev: i64 = runs[*pos - 1];
    let curr: i64 = runs[*pos];
    let next: i64 = runs[*pos + 1];

    if curr == k as i64 && next == k as i64 {
	res[*pos] = 'M';
    } else if curr > threshold as i64 && (next > 0 && next < threshold as i64) {
	res[*pos] = 'R';
	res[*pos + 1] = 'R';
    } else if next == 1 && curr == curr_ms as i64 {
	res[*pos] = 'M';
    } else if curr > threshold as i64 {
	res[*pos] = 'M';
    } else if curr == next - 1 && curr > 0 {
	res[*pos] = 'M';
    } else if curr == 0 && next == 1 && prev > 0 {
	res[*pos] = 'X';
	res[*pos - 1] = 'M';
    } else if curr == 0 && next == 1 && prev == -1 {
	let mut next_gap: usize = pos.clone();
	let mut gap_len: usize = 0;
	while runs[next_gap - 1] < 0 && next_gap > 1 {
	    gap_len += 1;
	    next_gap -= 1;
	}
	// TODO Determine what counts as an insertion or gap in run_to_aln
	while *pos < *pos + gap_len && *pos < runs.len() {
	    res[*pos] = if gap_len > 29 { '-' } else { 'I' };
	    *pos += 1;
	}
    } else {
	res[*pos] = ' ';
    };
}

pub fn derandomize_ms(
    ms: &Vec<usize>,
    params_in: &Option<TranslateParams>
) -> Vec<i64> {
    let params = params_in.clone().unwrap();
    let len = ms.len();

    let mut runs: Vec<i64> = vec![0; len];

    // Traverse the matching statistics in reverse
    runs[len - 1] = ms[len - 1] as i64;
    for i in 2..len {
	runs[len - i] = ms_to_run(ms[len - i], ms[len - i + 1], runs[len - i + 1], params.threshold, params.k);
    }

    return runs;
}


pub fn translate_runs(
    ms: &Vec<usize>,
    runs: &Vec<i64>,
    params_in: &Option<TranslateParams>,
) -> Vec<char> {
    let params = params_in.clone().unwrap();
    let len = runs.len();
    let mut aln = vec![' '; len];

    // Traverse the runs
    for mut i in 3..(len - 1) {
	run_to_aln(&runs, ms[i], params.threshold, params.k, &mut aln, &mut i);
    }

    return aln;
}

pub fn run_lengths(
    aln: &Vec<char>,
) -> Vec<(usize, usize, usize, usize)> {
    // Store run lengths as Vec<(start, end, matches, mismatches)>
    let mut encodings: Vec<(usize, usize, usize, usize)> = Vec::new();

    let mut i = 0;
    let mut match_start: bool = false;
    while i < aln.len() {
	match_start = (aln[i] != '-' && aln[i] != ' ') && !match_start;
	if match_start {
	    let start = i;
	    let mut matches: usize = 0;
	    while i < aln.len() && (aln[i] != '-' && aln[i] != ' ') {
		matches += (aln[i] == 'M' || aln[i] == 'R') as usize;
		i += 1;
	    }
	    encodings.push((start + 1, i, matches, i - start - matches));
	    match_start = false;
	} else {
	    i += 1;
	}
    }
    return encodings;
}

////////////////////////////////////////////////////////////////////////////////
// Tests
//
#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn log_rm_max_cdf() {
	let expected = vec![-1306319.1078024083,-318761.2492719044,-79220.9269610741,-19776.1823255263,-4942.2344281681,-1235.4454790664,-308.8543003470,-77.2131332649,-19.3032557026,-4.8258121998,-1.2064529421,-0.3016132288,-0.0754033068,-0.0188508267,-0.0047127067,-0.0011781767,-0.0002945442,-0.0000736360,-0.0000184090,-0.0000046023,-0.0000011506,-0.0000002876,-0.0000000719,-0.0000000180,-0.0000000045,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000];
	let alphabet_size = 4;
	let n_kmers = 20240921;
	let k = 1..32;
	k.for_each(|t| assert_approx_eq!(super::log_rm_max_cdf(t, alphabet_size, n_kmers), expected[t - 1], 1e-8f64));
    }

    #[test]
    fn random_match_threshold() {
	let expected = vec![15,18,22,25,28];
	let alphabet_size = 4;
	let n_kmers = 20240921;
	let k = 31;
	let factor = 1..6;
	factor.for_each(|i| assert_eq!(super::random_match_threshold(k, n_kmers, alphabet_size, (0.01_f64).powf(i as f64)), expected[i - 1]));
    }

    // TODO Test cases for ms_to_run

    // TODO Test cases for run_to_aln

    #[test]
    fn derandomize_ms() {
	let input = vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,11,11,12,11,10,11,11,12,11,12,10,11,12,12,10,11,11,11,11,11,11,10,11,11,12,13,11,12,13,14,15,16,13,14,15,16,12,12,13,14,15,16,17,18,19,20,21,22,12,10,10,11,12,11,10,11,12,11,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,13,14,15,12,12,10,11,11,11,12,13,13,14,15,11,11,11,11,11,11,12,13,14,11,11,11,11,12,13,12,12,12,12,13,12,13,14,12,13,11,12,12,11,12,11,12,13,14,14,13,14,15,15,16,17,18,19,19,19,20,21,22,12,13,11,11,12,12,13,14,15,16,17,18,19,20,21,22,10,11,9,10,10,11,11,12,11,11,12,13,13,14,12,11,11,12,13,12,13,12,12,12,12,13,11,12,12,10,11,11,10,11,11,12,10,9,10,10,10,11,12,10,9,10,10,10,11,10,11,12,10,8,9,10,9,9,10,9,10,10,10,11,12,13,14,15,16,17,13,11,11,11,12,11,11,12,12,11,11,12,12,13,14,15,11,12,10,11,9,10,11,11,11,11,11,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,11,12,12,13,11,11,12,13,14,13,11,11,12,13,14,15,16,17,18,19,20,21,11,12,11,11,12,11,12,12,12,12,11,10,11,12,11,11,12,13,12,12,11,12,13,13,13,11,11,12,11,12,13,12,13,14,15,16,17,18,19,20,21,11,12,13,9,10,11,10,10,10,11,12,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27];
	let expected = vec![0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,-164,-163,-162,-161,-160,-159,-158,-157,-156,-155,-154,-153,-152,-151,-150,-149,-148,-147,-146,-145,-144,-143,-142,-141,-140,-139,-138,-137,-136,-135,-134,-133,-132,-131,-130,-129,-128,-127,-126,-125,-124,-123,-122,-121,-120,-119,-118,-117,-116,-115,-114,-113,-112,-111,-110,-109,-108,-107,-106,-105,-104,-103,-102,-101,-100,-99,-98,-97,-96,-95,-94,-93,-92,-91,-90,-89,-88,-87,-86,-85,-84,-83,-82,-81,-80,-79,-78,-77,-76,-75,-74,-73,-72,-71,-70,-69,-68,-67,-66,-65,-64,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49,-48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49,-48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27];

	let got = super::derandomize_ms(&input, &Some(super::TranslateParams{ k: 31, threshold: 22 }));
	assert_eq!(got, expected);
    }

    // TODO Fix test for translate_runs
    // #[test]
    // fn translate_runs() {
    // 	let expected = vec!['-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M'];
    // 	let input_ms: Vec<usize> = vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,11,11,12,11,10,11,11,12,11,12,10,11,12,12,10,11,11,11,11,11,11,10,11,11,12,13,11,12,13,14,15,16,13,14,15,16,12,12,13,14,15,16,17,18,19,20,21,22,12,10,10,11,12,11,10,11,12,11,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,13,14,15,12,12,10,11,11,11,12,13,13,14,15,11,11,11,11,11,11,12,13,14,11,11,11,11,12,13,12,12,12,12,13,12,13,14,12,13,11,12,12,11,12,11,12,13,14,14,13,14,15,15,16,17,18,19,19,19,20,21,22,12,13,11,11,12,12,13,14,15,16,17,18,19,20,21,22,10,11,9,10,10,11,11,12,11,11,12,13,13,14,12,11,11,12,13,12,13,12,12,12,12,13,11,12,12,10,11,11,10,11,11,12,10,9,10,10,10,11,12,10,9,10,10,10,11,10,11,12,10,8,9,10,9,9,10,9,10,10,10,11,12,13,14,15,16,17,13,11,11,11,12,11,11,12,12,11,11,12,12,13,14,15,11,12,10,11,9,10,11,11,11,11,11,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,11,12,12,13,11,11,12,13,14,13,11,11,12,13,14,15,16,17,18,19,20,21,11,12,11,11,12,11,12,12,12,12,11,10,11,12,11,11,12,13,12,12,11,12,13,13,13,11,11,12,11,12,13,12,13,14,15,16,17,18,19,20,21,11,12,13,9,10,11,10,10,10,11,12,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27];
    // 	let input_runs: Vec<i64> = vec![0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,-164,-163,-162,-161,-160,-159,-158,-157,-156,-155,-154,-153,-152,-151,-150,-149,-148,-147,-146,-145,-144,-143,-142,-141,-140,-139,-138,-137,-136,-135,-134,-133,-132,-131,-130,-129,-128,-127,-126,-125,-124,-123,-122,-121,-120,-119,-118,-117,-116,-115,-114,-113,-112,-111,-110,-109,-108,-107,-106,-105,-104,-103,-102,-101,-100,-99,-98,-97,-96,-95,-94,-93,-92,-91,-90,-89,-88,-87,-86,-85,-84,-83,-82,-81,-80,-79,-78,-77,-76,-75,-74,-73,-72,-71,-70,-69,-68,-67,-66,-65,-64,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49,-48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49,-48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27];
    // 	let got = super::translate_runs(&input_ms, &input_runs, &Some(super::TranslateParams{ k: 31, threshold: 22 }));
    // 	assert_eq!(got, expected);
    // }

    #[test]
    fn run_lengths() {
	let expected: Vec<(usize, usize, usize, usize)> = vec![(6,33,28,0),(82,207,126,0),(373,423,51,0),(488,512,25,0)];
	let input = vec!['-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M'];
	let got = super::run_lengths(&input);
	assert_eq!(got, expected);
    }
}
