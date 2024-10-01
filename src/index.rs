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
//! Wrapper for using the [sbwt](https://docs.rs/sbwt) API to build and query SBWT indexes.
use std::ffi::OsString;
use std::io::Write;
use std::ops::Range;
use std::path::PathBuf;

use sbwt::BitPackedKmerSorting;
use sbwt::SbwtIndexBuilder;
use sbwt::SbwtIndexVariant;

/// Controls the parameters and resources available to the SBWT construction algorithm.
///
/// Used to specify values for:
/// - _k_-mer size `k`.
/// - Reverse complement input sequences `add_revcomp`.
/// - Number of threads `num_threads` to use.
/// - Size of the precalculated lookup table `prefix_precalc`.
/// - Build select support `build_select` (required for [map]).
/// - RAM available (in GB) to construction algorithm `mem_gb`.
/// - Deduplicate _k_-mer batches in construction algorithm `dedup_batches`.
/// - Temporary directory path `temp_dir`.
///
/// Note that this struct is marked non_exhaustive, meaning that new
/// public fields may be added in the future.
///
/// Same as initializing manually with these values:
/// ```rust
/// let mut opts = sablast::index::BuildOpts::default();
/// opts.k = 31;
/// opts.add_revcomp = false;
/// opts.num_threads = 1;
/// opts.prefix_precalc = 8;
/// opts.build_select = false;
/// opts.mem_gb = 4;
/// opts.dedup_batches = false;
/// opts.temp_dir = None;
///
/// # let expected = sablast::index::BuildOpts::default();
/// # assert_eq!(opts.k, expected.k);
/// # assert_eq!(opts.add_revcomp, expected.add_revcomp);
/// # assert_eq!(opts.num_threads, expected.num_threads);
/// # assert_eq!(opts.prefix_precalc, expected.prefix_precalc);
/// # assert_eq!(opts.build_select, expected.build_select);
/// # assert_eq!(opts.mem_gb, expected.mem_gb);
/// # assert_eq!(opts.dedup_batches, expected.dedup_batches);
/// # assert_eq!(opts.temp_dir, expected.temp_dir);
/// ```
///
#[derive(Clone)]
#[non_exhaustive]
pub struct BuildOpts {
    pub k: usize,
    pub add_revcomp: bool,
    pub num_threads: usize,
    pub prefix_precalc: usize,
    pub build_select: bool,
    pub mem_gb: usize,
    pub dedup_batches: bool,
    pub temp_dir: Option<String>,
}
// Defaults
impl Default for BuildOpts {
    fn default() -> BuildOpts {
        BuildOpts {
	    k: 31,
	    add_revcomp: false,
	    num_threads: 1,
	    prefix_precalc: 8,
	    build_select: false,
	    mem_gb: 4,
	    dedup_batches: false,
	    temp_dir: None,
        }
    }
}

/// Builds an SBWT index and its LCS array from sequences in memory.
///
/// Passes all character sequences in `slices` to the SBWT API calls
/// to build the SBWT index and LCS array. Use the [BuildOpts]
/// argument `build_options` to control the options and resources
/// passed to the index builder.
///
/// Note this function considers all data in `slices` as belonging to
/// the same sequence, meaning that only one index will be built.
///
/// Returns a tuple containing the SBWT index and the LCS array.
///
/// Requires write access to some temporary directory. Path can be set
/// using temp_dir in BuildOpts; defaults to $TMPDIR on Unix if not set.
///
/// # Examples
/// ```rust
/// use sablast::index::*;
///
/// // Inputs
/// let reference: Vec<Vec<u8>> = vec![vec![b'A',b'A',b'A',b'G',b'A',b'A',b'C',b'C',b'A',b'-',b'T',b'C',b'A',b'G',b'G',b'G',b'C',b'G']];
///
/// // Build the SBWT
/// let mut opts = BuildOpts::default();
/// opts.k = 3;
/// let (sbwt, lcs) = build_sbwt_from_vecs(&reference, &Some(opts));
/// ```
///
pub fn build_sbwt_from_vecs(
    slices: &[Vec<u8>],
    build_options: &Option<BuildOpts>,
) -> (sbwt::SbwtIndexVariant, sbwt::LcsArray) {
    assert!(!slices.is_empty());

    let build_opts = if build_options.is_some() { build_options.clone().unwrap() } else { BuildOpts::default() };
    let temp_dir = if build_opts.temp_dir.is_some() { build_opts.temp_dir.unwrap() } else { std::env::temp_dir().to_str().unwrap().to_string() };

    let algorithm = BitPackedKmerSorting::new()
	.mem_gb(build_opts.mem_gb)
	.dedup_batches(build_opts.dedup_batches)
	.temp_dir(PathBuf::from(OsString::from(temp_dir)).as_path());

    let (sbwt, lcs) = SbwtIndexBuilder::new()
	.k(build_opts.k)
	.n_threads(build_opts.num_threads)
	.add_rev_comp(build_opts.add_revcomp)
	.algorithm(algorithm)
	.build_lcs(true)
	.build_select_support(build_opts.build_select)
	.precalc_length(build_opts.prefix_precalc)
	.run_from_vecs(slices);

    (SbwtIndexVariant::SubsetMatrix(sbwt), lcs.unwrap())
}

/// Writes an SBWT index and its LCS array to disk.
///
/// Creates the files `outfile_prefix` + ".sbwt" and `outfile_prefix` +
/// ".lcs" to store the SBWT index `sbwt` and the LCS array `lcs`.
///
/// Panics if the output files cannot be created with
/// std::fs::File::create or are not writable by
/// std::io::BufWriter::new.
///
/// # Examples
/// ```rust
/// use sablast::index::*;
///
/// // Inputs
/// let reference: Vec<Vec<u8>> = vec![vec![b'A',b'A',b'A',b'G',b'A',b'A',b'C',b'C',b'A',b'-',b'T',b'C',b'A',b'G',b'G',b'G',b'C',b'G']];
///
/// // Build the SBWT
/// let mut opts = BuildOpts::default();
/// opts.k = 3;
/// let (sbwt, lcs) = build_sbwt_from_vecs(&reference, &Some(opts));
///
/// // Serialize the sbwt to $TMPDIR/serialized_index_1
/// let index_prefix = std::env::temp_dir().to_str().unwrap().to_owned() + "/serialized_index_1";
/// serialize_sbwt(&index_prefix, &sbwt, &lcs);
/// ```
///
pub fn serialize_sbwt(
    outfile_prefix: &str,
    sbwt: &sbwt::SbwtIndexVariant,
    lcs: &sbwt::LcsArray,
) {
    let sbwt_outfile = format!("{}.sbwt", outfile_prefix);
    let lcs_outfile = format!("{}.lcs", outfile_prefix);

    // Write sbwt
    let sbwt_conn = std::fs::File::create(&sbwt_outfile).unwrap_or_else(|_| panic!("Expected write access to {}", sbwt_outfile));
    let mut sbwt_out = std::io::BufWriter::new(sbwt_conn);
    sbwt_out.write_all(&(b"SubsetMatrix".len() as u64).to_le_bytes()).expect("Serialized SBWT header part 1.");
    sbwt_out.write_all(b"SubsetMatrix").expect("Serialized SBWT header part 2.");
    match sbwt {
        SbwtIndexVariant::SubsetMatrix(index) => {
	    index.serialize(&mut sbwt_out).expect("Serialized SBWT index.");
	},
    };

    // Write lcs array
    let lcs_conn = std::fs::File::create(&lcs_outfile).unwrap_or_else(|_| panic!("Expected write access to {}", lcs_outfile));
    let mut lcs_out = std::io::BufWriter::new(lcs_conn);
    lcs.serialize(&mut lcs_out).expect("Serialized LCS array.");
}

/// Loads a prebuilt SBWT index and its LCS array from disk.
///
/// Reads the SBWT index stored at `index_prefix` + ".sbwt" and the
/// LCS array at `index_prefix` + ".lcs".
///
/// Returns a tuple containing the SBWT index variant and the LCS
/// array.
///
/// Panics if the SBWT or the LCS file are not readable with
/// std::fs::File::open.
///
/// # Examples
/// ```rust
/// use sablast::index::*;
///
/// // Inputs
/// let reference: Vec<Vec<u8>> = vec![vec![b'A',b'A',b'A',b'G',b'A',b'A',b'C',b'C',b'A',b'-',b'T',b'C',b'A',b'G',b'G',b'G',b'C',b'G']];
///
/// // Build the SBWT
/// let mut opts = BuildOpts::default();
/// opts.k = 3;
/// let (sbwt, lcs) = build_sbwt_from_vecs(&reference, &Some(opts));
///
/// // Serialize the sbwt to $TMPDIR/serialized_index_2
/// let index_prefix = std::env::temp_dir().to_str().unwrap().to_owned() + "/serialized_index_2";
/// serialize_sbwt(&index_prefix, &sbwt, &lcs);
///
/// // Load index
/// let (sbwt_loaded, lcs_loaded) = load_sbwt(&index_prefix);
/// # assert_eq!(lcs, lcs_loaded);
/// # match sbwt_loaded {
/// #     sbwt::SbwtIndexVariant::SubsetMatrix(ref loaded) => {
/// #         match sbwt_loaded {
/// #             sbwt::SbwtIndexVariant::SubsetMatrix(ref built) => {
/// #                 assert_eq!(built, loaded);
/// #             },
/// #         };
/// #     }
/// # }
/// ```
///
pub fn load_sbwt(
    index_prefix: &str,
) -> (sbwt::SbwtIndexVariant, sbwt::LcsArray) {
    let indexfile = format!("{}.sbwt", index_prefix);
    let lcsfile = format!("{}.lcs", index_prefix);

    // Load sbwt
    let sbwt_conn = std::fs::File::open(&indexfile).unwrap_or_else(|_| panic!("Expected SBWT at {}", indexfile));
    let mut index_reader = std::io::BufReader::new(sbwt_conn);
    let sbwt = sbwt::load_sbwt_index_variant(&mut index_reader).unwrap();

    // Load the lcs array
    let lcs_conn = std::fs::File::open(&lcsfile).unwrap_or_else(|_| panic!("Expected LCS array at {}", lcsfile));
    let mut lcs_reader = std::io::BufReader::new(lcs_conn);
    let lcs = sbwt::LcsArray::load(&mut lcs_reader).unwrap();

    (sbwt, lcs)
}

/// Queries an SBWT index for the _k_-bounded matching statistics.
///
/// Matches the _k_-mers in `query` against the SBWT index `index` and
/// its longest common suffix array `lcs`.
///
/// Returns a vector containing tuples with the _k_-bounded matching
/// statistic at the position of each element in the query and the
/// [colex interval](https://docs.rs/sbwt/latest/sbwt/) of the match.
///
/// # Examples
/// ```rust
/// use sablast::index::*;
///
/// // Inputs
/// let reference: Vec<Vec<u8>> = vec![vec![b'A',b'A',b'A',b'G',b'A',b'A',b'C',b'C',b'A',b'-',b'T',b'C',b'A',b'G',b'G',b'G',b'C',b'G']];
/// let query: Vec<u8> = vec![b'C',b'A',b'A',b'G',b'C',b'C',b'A',b'C',b'T',b'C',b'A',b'T',b'T',b'G',b'G',b'G',b'T',b'C'];
///
/// // Build the SBWT
/// let mut opts = BuildOpts::default();
/// opts.k = 3;
/// let (sbwt, lcs) = build_sbwt_from_vecs(&reference, &Some(opts));
///
/// // Run query
/// let ms: Vec<usize> = query_sbwt(&query, &sbwt, &lcs).iter().map(|x| x.0).collect();
/// // `ms` has [1,2,2,3,2,2,3,2,1,2,3,1,1,1,2,3,1,2]
/// # assert_eq!(ms, vec![1,2,2,3,2,2,3,2,1,2,3,1,1,1,2,3,1,2]);
/// ```
///
pub fn query_sbwt(
    query: &[u8],
    sbwt: &sbwt::SbwtIndexVariant,
    lcs: &sbwt::LcsArray,
) -> Vec<(usize, Range<usize>)> {
    assert!(!query.is_empty());
    let ms = match sbwt {
        SbwtIndexVariant::SubsetMatrix(index) => {
	    let streaming_index = sbwt::StreamingIndex::new(index, lcs);
	    streaming_index.matching_statistics(query)
	},
    };
    ms
}

////////////////////////////////////////////////////////////////////////////////
// Tests
//
#[cfg(test)]
mod tests {
    #[test]
    fn build_and_query_sbwt() {
	let reference: Vec<Vec<u8>> = vec![vec![b'A',b'A',b'A',b'G',b'A',b'A',b'C',b'C',b'A',b'-',b'T',b'C',b'A',b'G',b'G',b'G',b'C',b'G']];
	let query: Vec<u8> = vec![b'C',b'A',b'A',b'G',b'C',b'C',b'A',b'C',b'T',b'C',b'A',b'T',b'T',b'G',b'G',b'G',b'T',b'C'];

	let (sbwt, lcs) = super::build_sbwt_from_vecs(&reference, &Some(super::BuildOpts{ k: 3, ..Default::default() }));

	let expected = vec![1,2,2,3,2,2,3,2,1,2,3,1,1,1,2,3,1,2];
	let got: Vec<usize> = super::query_sbwt(&query, &sbwt, &lcs).iter().map(|x| x.0).collect();

	assert_eq!(got, expected);
    }

    #[test]
    fn build_serialize_load_sbwt() {
	let reference: Vec<Vec<u8>> = vec![vec![b'A',b'A',b'A',b'G',b'A',b'A',b'C',b'C',b'A',b'-',b'T',b'C',b'A',b'G',b'G',b'G',b'C',b'G']];
	let (sbwt, lcs) = super::build_sbwt_from_vecs(&reference, &Some(super::BuildOpts{ k: 3, ..Default::default() }));

	let index_prefix = std::env::temp_dir().to_str().unwrap().to_owned() + "/serialized_index_test";
	super::serialize_sbwt(&index_prefix, &sbwt, &lcs);

	let (sbwt_loaded, lcs_loaded) = super::load_sbwt(&index_prefix);

	assert_eq!(lcs, lcs_loaded);
	match sbwt {
            sbwt::SbwtIndexVariant::SubsetMatrix(ref index) => {
		match sbwt_loaded {
		    sbwt::SbwtIndexVariant::SubsetMatrix(ref index_loaded) => {
			assert_eq!(index, index_loaded);
		    },
		};
	    },
	};
    }
}
