import sys
sys.path.append(".")
from pyrouge import Rouge155
import os
from src.data_until import config


def caculate_rouge_score(_rouge_ref_dir, _rouge_dec_dir):
    r = Rouge155()
    r.system_dir = _rouge_ref_dir
    r.model_dir = _rouge_dec_dir
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_filename_pattern = '#ID#_reference.txt
    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    rouge_log(output_dict, config.root_dir)

def rouge_log(results_dict, dir_to_write):
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str)
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
  print("Writing final ROUGE results to %s..."%(results_file))
  with open(results_file, "w") as f:
    f.write(log_str)

caculate_rouge_score('decoded/', 'reference/')
