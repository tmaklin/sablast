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
//! Translating deterministic _k_-bounded matching statistics into alignments.
//!
//! The translated alignment is encoded using the following characters:
//! - **M** : Match between query and reference.
//! - **-** : Characters in the query that are not found in the reference.
//! - **X** : Single character mismatch or insertion into the query.
//! - **R** : Two consecutive 'R's signify a discontinuity in the alignment.
//!           The right 'R' is at the start of a _k_-mer that is not adjacent
//!           to the last character in the _k_-mer corresponding to the left
//!           'R'. This implies either a deletion of unknown length in the query,
//!           or insertion of _k_-mers from elsewhere in the reference into the query.
//!
use std::ops::Range;

use sbwt::SbwtIndexVariant;
use sbwt::SubsetSeq;

/// Translates a single derandomized _k_-bounded matching statistic.
///
/// Translates the current derandomized matching statistic (MS)
/// `ms_curr` based on the values of its left `ms_prev` and right
/// `ms_next` neighbors and the lower bound `threshold` for random
/// matches.
///
/// Returns a tuple containing the translation of the current MS and
/// translation of the right neighbor match. The right neighbor is an
/// empty string literal ' ' if translation of the current MS does not
/// affect its value.
///
/// # Examples
/// ## Query with only matches
/// ```rust
/// use sablast::translate::translate_ms_val;
///
/// // Parameters       : k = 3, threshold = 2
/// //
/// // Ref sequence     : A,C,G,C,A,G
/// // Query sequence   : A,C,G,C,A,G
/// //
/// // Result MS vector : 1,2,3,3,3,3
/// // Testing this pos :     |
/// // Expected output  : M,M,M,M,M,M
///
/// let translated = translate_ms_val(1, 2, 3, 2);
/// // `translated` has ('M', ' ')
/// # assert_eq!(translated.0, 'M');
/// # assert_eq!(translated.1, ' ');
/// ```
///
/// ## Query with a single mismatch
/// ```rust
/// use sablast::translate::translate_ms_val;
///
/// // Parameters       : k = 3, threshold = 2
/// //
/// // Ref sequence     : A,C,G,T,C,A,G
/// // Query sequence   : A,C,G,C,C,A,G
/// //
/// // Result MS vector : 1,2,3,0,1,2,3
/// // Testing this pos :       |
/// // Expected output  : M,M,M,X,M,M,M
///
/// let translated = translate_ms_val(0, 1, 3, 2);
/// // `translated` has ('X', ' ')
/// # assert_eq!(translated.0, 'X');
/// # assert_eq!(translated.1, ' ');
/// ```
///
/// ## Query with a single insertion:
/// ```rust
/// use sablast::translate::translate_ms_val;
///
/// // Ref sequence     : A,C,G,-,C,A,G
/// // Query sequence   : A,C,G,C,C,A,G
/// //
/// // Result MS vector : 1,2,3,0,1,2,3
/// // Testing this pos :       |
/// // Expected output  : M,M,M,X,M,M,M
///
/// let translated = translate_ms_val(0, 1, 3, 2);
/// // `translated` has ('X', ' ')
/// # assert_eq!(translated.0, 'X');
/// # assert_eq!(translated.1, ' ');
/// ```
///
/// Note that this case is identical to the query with a single
/// mismatch. These two are indistinguishible based on the _k_-bounded
/// matching statistics alone although the input sequences are
/// different.
///
/// ## Query with multiple insertions:
/// ```rust
/// use sablast::translate::translate_ms_val;
///
/// // Ref sequence     : A,C,G, -,-,C,A,G
/// // Query sequence   : A,C,G, T,T,C,C,A,G
/// //
/// // Result MS vector : 1,2,3,-1,0,1,2,3
/// // Testing this pos :        |
/// // Expected output  : M,M,M, -,-,M,M,M
///
/// let translated = translate_ms_val(-1, 0, 3, 2);
/// // `translated` has ('-', ' ')
/// # assert_eq!(translated.0, '-');
/// # assert_eq!(translated.1, ' ');
///
/// ```
/// ## Query with a deletion
/// ```rust
/// use sablast::translate::translate_ms_val;
///
///
/// // Parameters       : k = 3, threshold = 2
/// //
/// // Ref sequence     : A,C,G,T,T,T,C,A,G
/// // Query sequence   : A,C,G,-,-,-,C,A,G
/// //
/// // Result MS vector : 1,2,3,1,2,3
/// // Testing this pos :     |
/// // Expected output  : M,M,R,R,M,M
///
/// let translated = translate_ms_val(3, 1, 2, 2);
/// // `translated` has ('R', 'R')
/// # assert_eq!(translated.0, 'R');
/// # assert_eq!(translated.1, 'R');
/// ```
///
/// Although in this case two characters have been deleted from the
/// query, if the query extends beyond what is shown the matching
/// statistics (MS) could also represent recombination of a sequence
/// from elsewhere in the query into the position preceding the three
/// consecutive T's in the reference.
///
/// ## Query with recombination
/// ```rust
/// use sablast::translate::translate_ms_val;
///
/// // Parameters       : k = 3, threshold = 2
/// //
/// // Ref sequence     : A,C,G,T,T,T,C,G,G,C,C,C
/// // Query sequence   : A,C,G,C,G,G,T,T,T,C,C,C
/// //
/// // Result MS vector : 1,2,3,1,2,3,3,3,3,1,2,3
/// // Testing this pos :                 |
/// // Expected output  : M,M,R,R,M,M,M,M,R,R,M,M
///
/// let translated = translate_ms_val(3, 1, 3, 2);
/// // `translated` has ('R', 'R')
/// # assert_eq!(translated.0, 'R');
/// # assert_eq!(translated.1, 'R');
/// ```
///
/// Note how the two regions with the consecutive 'R's are similar to
/// the Query with a deletion case. The first R,R pair is exactly the
/// same, while the second R,R pair is only different because the
/// match extends further to the left of it.
///
/// When a segment of reasonable length is encapsulated by two
/// consecutive R's on both the left and right side, the region in
/// between possibly originates from elsewhere in the reference.
///
/// In general recombinations and deletions are indistinguishable in
/// the _k_-bounded matching statistics alone but they can be solved
/// by comparing the MS vector with the reference and query.
///
pub fn translate_ms_val(
    ms_curr: i64,
    ms_next: i64,
    ms_prev: i64,
    threshold: usize,
) -> (char, char) {
    assert!(threshold > 1);

    let aln_curr: char;
    let mut aln_next: char = ' ';
    if ms_curr > threshold as i64 && ms_next > 0 && ms_next < threshold as i64 {
	// Current position is first character in a jump to another k-mer,
	// or there is deletion of unknown length in the query wrt. the reference.
	//
	// Use two consecutive 'R's to denote breakpoint between two k-mers
	aln_curr = 'R';
	aln_next = 'R';
    } else if ms_curr <= 0 {
	// Start of a mismatch region
	if ms_next == 1 && ms_prev > 0 {
	    // Mismatched character or insertion of 1 character in the query.
	    //
	    // Use 'X' for mismatch or 1 character insert
	    aln_curr = 'X';
	} else {
	    // Insertion of more than 1 characters in the query
	    //
	    // Use '-' to denote inserts of more than 1 characters
	    aln_curr = '-';
	}
    } else {
	// Other values are always a match, use 'M' for these
	aln_curr = 'M';
    }

    (aln_curr, aln_next)
}

/// Translates a sequence of derandomized _k_-bounded matching statistics.
///
/// Iterates over a derandomized sequence of _k_-bounded matching
/// statistics `derand_ms` for _k_-mers with size `k` derandomized
/// with the threshold `threshold`.
///
/// Returns a sequence containing a character representation of the
/// underlying alignment.
///
/// # Examples
/// ## Translate a generic MS vector
/// ```rust
/// use sablast::translate::translate_ms_vec;
///
/// // Parameters       : k = 3, threshold = 2
/// //
/// // Ref sequence     : A,A,A,G,A,A,C,C,A,-,T,C,A, -,-,G,G,G, C,G
/// // Query sequence   : C,A,A,G,-,-,C,C,A,C,T,C,A, T,T,G,G,G, T,C
/// // Input MS         : 0,1,2,3,    1,2,3,0,1,2,3,-1,0,1,2,3,-1,0
/// // Expected output  : X,M,M,R,    R,M,M,X,M,M,M, -,-,M,M,M, -,-
///
/// let input: Vec<i64> = vec![0,1,2,3,1,2,3,0,1,2,3,-1,0,1,2,3,-1,0];
/// let translated = translate_ms_vec(&input, 3, 2);
/// // `translated` has ['X','M','M','R','R','M','M','X','M','M','M','-','-','M','M','M','-','-']
/// # assert_eq!(translated, vec!['X','M','M','R','R','M','M','X','M','M','M','-','-','M','M','M','-','-']);
/// ```
///
/// ## Translate a MS vector with recombination
/// ```rust
/// use sablast::translate::translate_ms_vec;
///
/// // Parameters       : k = 3, threshold = 2
/// //
/// // Ref sequence     : A,C,G,T,T,T,C,G,G,C,C,C
/// // Query sequence   : A,C,G,C,G,G,T,T,T,C,C,C
/// //
/// // Result MS vector : 1,2,3,1,2,3,3,3,3,1,2,3
/// // Expected output  : M,M,R,R,M,M,M,M,R,R,M,M
///
/// let input: Vec<i64> = vec![1,2,3,1,2,3,3,3,3,1,2,3,];
/// let translated = translate_ms_vec(&input, 3, 2);
/// // `translated` has ['M','M','R','R','M','M','M','M','R','R','M','M']
/// # assert_eq!(translated, vec!['M','M','R','R','M','M','M','M','R','R','M','M']);
/// ```
///
pub fn translate_ms_vec(
    derand_ms: &[i64],
    k: usize,
    threshold: usize,
) -> Vec<char> {
    assert!(k > 0);
    assert!(threshold > 1);
    assert!(derand_ms.len() > 2);

    let len = derand_ms.len();
    let mut res = vec![' '; len];

    // Traverse the derandomized matching statistics
    for pos in 0..len {
	let prev: i64 = if pos > 1 { derand_ms[pos - 1] } else { 31 };
	let curr: i64 = derand_ms[pos];
	let next: i64 = if pos < len - 1 { derand_ms[pos + 1] } else { derand_ms[pos] };

	// Two consecutive 'R's mean this pos was already set by the previous iteration
	if !(pos > 1 && res[pos - 1] == 'R' && res[pos] == 'R') {
	    let (aln_curr, aln_next) = translate_ms_val(curr, next, prev, threshold);

	    res[pos] = aln_curr;
	    if pos + 1 < len - 1 && aln_next != ' ' {
		res[pos + 1] = aln_next;
	    }
	}
    }

    res
}

/// Refines a translated alignment by resolving SNPs.
///
/// Resolves all 'X's in the translation `translation` by using the
/// colexicographic intervals stored in the second component of
/// `noisy_ms` tuples to extract the _k_-mers that overlap the 'X's
/// from the SBWT index `query_sbwt`.
///
/// The 'X's are resolved by checking whether the _k_-mer whose first
/// character overlaps the 'X' has a matching statistic of _k_ - 1
/// and, if it does, replacing the character of 'X' with a character
/// from the _k_-mer that has 'X' as its middle base (a '[split
/// _k_-mer](https://docs.rs/ska). If the matching statistic is less
/// than _k_ - 1, the character is checked from the _k_-mer that is
/// (`threshold` + 1)/2 characters away from the 'X'.
///
/// The SBWT index must have [select
/// support](https://docs.rs/sbwt/latest/sbwt/struct.SbwtIndexBuilder.html)
/// enabled.
///
/// Returns a refined translation where the 'X's have been replaced
/// with the substituted character.
///
/// # Examples
/// ```rust
/// use sablast::build;
/// use sablast::index::BuildOpts;
/// use sablast::index::query_sbwt;
/// use sablast::derandomize::derandomize_ms_vec;
/// use sablast::derandomize::random_match_threshold;
/// use sablast::translate::translate_ms_vec;
/// use sablast::translate::refine_translation;
/// use sbwt::SbwtIndexVariant;
///
/// // Parameters       : k = 4, threshold = 3
/// //
/// // Ref sequence     : T,T,G,A, T,T,G,G,C,T,G,G,G,C,A,G,A,G,C,T,G
/// // Query sequence   : T,T,G,A,     G,G,C,T,G,G,G,G,A,G,A,G,C,T,G
/// //
/// // Result MS vector : 1,2,3,4, 1,2,3,3,3,4,4,4,4,3,1,2,3,4,4,4,4
/// // Derandomized MS  : 1,2,3,4,-1,0,1,2,3,4,4,4,4,0,1,2,3,4,4,4,4
/// // Translation      : M,M,M,M, -,-,M,M,M,M,M,M,M,X,M,M,M,M,M,M,M
/// // Refined          : M,M,M,M, -,-,M,M,M,M,M,M,M,G,M,M,M,M,M,M,M
/// // Changed this pos :                            |
///
/// let query: Vec<u8> = vec![b'T',b'T',b'G',b'A',b'G',b'G',b'C',b'T',b'G',b'G',b'G',b'G',b'A',b'G',b'A',b'G',b'C',b'T',b'G'];
/// let reference: Vec<u8> = vec![b'T',b'T',b'G',b'A',b'T',b'T',b'G',b'G',b'C',b'T',b'G',b'G',b'G',b'C',b'A',b'G',b'A',b'G',b'C',b'T',b'G'];
///
/// let mut opts = BuildOpts::default();
/// opts.k = 4;
/// opts.build_select = true;
/// let (sbwt, lcs) = build(&[query], opts);
///
/// let k = match sbwt {
///     SbwtIndexVariant::SubsetMatrix(ref sbwt) => {
///         sbwt.k()
///     },
/// };
/// let threshold = 3;
///
/// let noisy_ms = query_sbwt(&reference, &sbwt, &lcs);
/// let derand_ms = derandomize_ms_vec(&noisy_ms.iter().map(|x| x.0).collect::<Vec<usize>>(), k, threshold);
/// let translated = translate_ms_vec(&derand_ms, k, threshold);
///
/// let refined = refine_translation(&translated, &noisy_ms, &sbwt, threshold);
///
/// # let expected = vec!['M','M','M','M','-','-','M','M','M','M','M','M','M','G','M','M','M','M','M','M','M'];
/// # assert_eq!(refined, expected);
/// ```
///
pub fn refine_translation(
    translation: &[char],
    noisy_ms: &[(usize, Range<usize>)],
    query_sbwt: &sbwt::SbwtIndexVariant,
    threshold: usize,
) -> Vec<char> {
    let n_elements = translation.len();
    assert!(!translation.is_empty());
    assert!(translation.len() == noisy_ms.len());
    let k = match query_sbwt {
	SbwtIndexVariant::SubsetMatrix(ref sbwt) => {
	    assert!(sbwt.sbwt().has_select_support());
	    assert!(sbwt.k() > 0);
	    sbwt.k()
	},
    };

    // Could prove midpoint is the best guess using conditional probabilities?
    // This is (coincidentally?) similar to split k-mers

    let mut refined = translation.to_vec().clone();
    match query_sbwt {
	SbwtIndexVariant::SubsetMatrix(ref sbwt) => {
	    for i in 1..(refined.len() - threshold) {
		if refined[i - 1] == 'X' {
		    let midpoint = if i + k - 2 < n_elements && noisy_ms[i + k - 2].0 == k - 1 { k/2 } else { (threshold + 1)/2 };
		    refined[i - 1] = sbwt.access_kmer(noisy_ms[i + k - 2 - midpoint].1.start)[midpoint] as char;
		}
	    }
	},
    };
    refined
}

////////////////////////////////////////////////////////////////////////////////
// Tests
//
#[cfg(test)]
mod tests {
    // Test cases for translate_ms_val
    // Comments use '-' for characters that are not in a ref or query sequence
    #[test]
    fn translate_ms_val_with_deletion() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G,T,T,T,C,A,G
	// Query sequence   : A,C,G,-,-,-,C,A,G
	//
	// Result MS vector : 1,2,3,1,2,3
	// Testing this pos :     |
	// Expected output  : M,M,R,R,M,M

	let expected = ('R','R');
	let got = super::translate_ms_val(3, 1, 2, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn translate_ms_val_with_recombination() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G,T,T,T,C,G,G,C,C,C
	// Query sequence   : A,C,G,C,G,G,T,T,T,C,C,C
	//
	// Result MS vector : 1,2,3,1,2,3,3,3,3,1,2,3
	// Testing this pos :                 |
	// Expected output  : M,M,R,R,M,M,M,M,R,R,M,M

	let expected = ('R','R');
	let got = super::translate_ms_val(3, 1, 3, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn translate_ms_val_with_mismatch() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G,T,C,A,G
	// Query sequence   : A,C,G,C,C,A,G
	//
	// Result MS vector : 1,2,3,0,1,2,3
	// Testing this pos :       |
	// Expected output  : M,M,M,X,M,M,M

	let expected = ('X',' ');
	let got = super::translate_ms_val(0, 1, 3, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn translate_ms_val_with_single_insertion() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G,-,C,A,G
	// Query sequence   : A,C,G,C,C,A,G
	//
	// Result MS vector : 1,2,3,0,1,2,3
	// Testing this pos :       |
	// Expected output  : M,M,M,X,M,M,M

	// Note this is identical to translate_ms_with_mismatch, these
	// are indistinguishible in outputs but have different inputs.
	// Kept here as a demonstration.
	let expected = ('X', ' ');
	let got = super::translate_ms_val(0, 1, 3, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn translate_ms_val_with_many_insertions() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G, -,-,C,A,G
	// Query sequence   : A,C,G, T,T,C,C,A,G
	//
	// Result MS vector : 1,2,3,-1,0,1,2,3
	// Testing this pos :        |
	// Expected output  : M,M,M, -,-,M,M,M

	let expected = ('-', ' ');
	let got = super::translate_ms_val(-1, 0, 3, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn translate_ms_val_with_only_matches() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G,C,A,G
	// Query sequence   : A,C,G,C,A,G
	//
	// Result MS vector : 1,2,3,3,3,3
	// Testing this pos :     |
	// Expected output  : M,M,M,M,M,M

	let expected = ('M', ' ');
	let got = super::translate_ms_val(1, 2, 3, 2);

	assert_eq!(got, expected);
    }
    
    #[test]
    fn translate_ms_vec() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,A,A,G,A,A,C,C,A,-,T,C,A, -,-,G,G,G, C,G
	// Query sequence   : C,A,A,G,-,-,C,C,A,C,T,C,A, T,T,G,G,G, T,C
	//
	// Result MS vector : 0,1,2,3,    1,2,3,0,1,2,3,-1,0,1,2,3,-1,0
	// Expected output  : X,M,M,R,    R,M,M,X,M,M,M, -,-,M,M,M, -,-

	let input: Vec<i64> = vec![0,1,2,3,1,2,3,0,1,2,3,-1,0,1,2,3,-1,0];
	let expected: Vec<char> = vec!['X','M','M','R','R','M','M','X','M','M','M','-','-','M','M','M','-','-'];
	let got = super::translate_ms_vec(&input, 3, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn translate_ms_vec_with_recombination() {
	// Parameters       : k = 3, threshold = 2
	//
	// Ref sequence     : A,C,G,T,T,T,C,G,G,C,C,C
	// Query sequence   : A,C,G,C,G,G,T,T,T,C,C,C
	//
	// Result MS vector : 1,2,3,1,2,3,3,3,3,1,2,3
	// Expected output  : M,M,R,R,M,M,M,M,R,R,M,M

	let input: Vec<i64> = vec![1,2,3,1,2,3,3,3,3,1,2,3];
	let expected: Vec<char> = vec!['M','M','R','R','M','M','M','M','R','R','M','M'];
	let got = super::translate_ms_vec(&input, 3, 2);

	assert_eq!(got, expected);
    }

    #[test]
    fn refine_translation() {
	use crate::build;
	use crate::index::BuildOpts;
	use crate::index::query_sbwt;
	use crate::derandomize::derandomize_ms_vec;
	use crate::derandomize::random_match_threshold;
	use super::translate_ms_vec;
	use super::refine_translation;
	use sbwt::SbwtIndexVariant;

	// Parameters       : k = 4, threshold = 3
	//
	// Ref sequence     : T,T,G,A, T,T,G,G,C,T,G,G,G,C,A,G,A,G,C,T,G
	// Query sequence   : T,T,G,A,     G,G,C,T,G,G,G,G,A,G,A,G,C,T,G
	//
	// Result MS vector : 1,2,3,4, 1,2,3,3,3,4,4,4,4,3,1,2,3,4,4,4,4
	// Derandomized MS  : 1,2,3,4,-1,0,1,2,3,4,4,4,4,0,1,2,3,4,4,4,4
	// Translation      : M,M,M,M, -,-,M,M,M,M,M,M,M,X,M,M,M,M,M,M,M
	// Refined          : M,M,M,M, -,-,M,M,M,M,M,M,M,G,M,M,M,M,M,M,M
	// Changed this pos :                            |

	let query: Vec<u8> = vec![b'T',b'T',b'G',b'A',b'G',b'G',b'C',b'T',b'G',b'G',b'G',b'G',b'A',b'G',b'A',b'G',b'C',b'T',b'G'];
	let reference: Vec<u8> = vec![b'T',b'T',b'G',b'A',b'T',b'T',b'G',b'G',b'C',b'T',b'G',b'G',b'G',b'C',b'A',b'G',b'A',b'G',b'C',b'T',b'G'];

	let (sbwt, lcs) = build(&[query], BuildOpts{ k: 4, build_select: true, ..Default::default() });

	let k = match sbwt {
	    SbwtIndexVariant::SubsetMatrix(ref sbwt) => {
		sbwt.k()
	    },
	};
	let threshold = 3;

	let noisy_ms = query_sbwt(&reference, &sbwt, &lcs);
	let derand_ms = derandomize_ms_vec(&noisy_ms.iter().map(|x| x.0).collect::<Vec<usize>>(), k, threshold);
	let translated = translate_ms_vec(&derand_ms, k, threshold);

	let refined = refine_translation(&translated, &noisy_ms, &sbwt, threshold);

	let expected = vec!['M','M','M','M','-','-','M','M','M','M','M','M','M','G','M','M','M','M','M','M','M'];
	assert_eq!(refined, expected);
    }
}
