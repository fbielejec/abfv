1. Input: two protein sequences for examples  (EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS, an antibody heavy chain - WGQGTLVTVSS) and (DIQMTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK, a light chain - FGQGTKVEIK), i.e. an antibody Fv.
2. Generate a 3D structure from the sequences, e.g. with ABodyBuilder3 (https://github.com/Exscientia/abodybuilder3).
3. Use FreeSASA (https://github.com/mittinatten/freesasa) to determine which amino acids of the chains are in contact.
4. Contact definition: residues whose side-chain surface accessibility (SASA) changes by at least 10% in the complex vs. computed for the isolated chains.
5. Visualize the results.
6. Wrap the whole thing in Docker and push to GitHub.
