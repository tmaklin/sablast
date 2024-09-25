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
#[test]
fn map_nissle_against_clbs() {
    use needletail::Sequence;

    let (sbwt, lcs) = sablast::index::build_sbwt_from_file(&"tests/data/clbS.fna.gz".to_string(), &None);

    let expected = vec![(455, 967, '+', 513, 1)];

    let mut reader = needletail::parse_fastx_file("tests/data/NZ_CP058217.1_clbS.fna.gz".to_string()).expect("valid path/file");
    let Some(rec) = reader.next() else { panic!("Couldn't read from tests/data/NZ_CP058217.1_clbS.fna.gz") };
    let seqrec = rec.expect("Valid fastX record");
    let seq = seqrec.normalize(true);

    let got = sablast::find(&seq, &sbwt, &lcs);

    assert_eq!(got, expected);
}
