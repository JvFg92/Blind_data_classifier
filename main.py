import data_pretreatment as dp
import mlp

if __name__ == "__main__":
  net=mlp.mlp_classifier(file_path=dp.find_best())
