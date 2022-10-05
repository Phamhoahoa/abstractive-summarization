import os

def write_for_rouge(reference_list, decoded_list,_rouge_ref_dir, _rouge_dec_dir):
    ref_file = os.path.join(_rouge_ref_dir, "reference.txt")
    decoded_file = os.path.join(_rouge_dec_dir, "decoded.txt")

    with open(ref_file, "w") as f:
        for line in reference_list:
            # write line to output file
            f.write(line)
            f.write("\n")
        f.close()

    with open(decoded_file, "w") as f:
        for line in decoded_list:
            # write line to output file
            f.write(line)
            f.write("\n")
        f.close()

# import files2rouge
# files2rouge.run(hyp_path, ref_path)
