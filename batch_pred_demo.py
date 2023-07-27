from src.pipeline.batch_prediction import strat_batch_prediction


file_path = "/home/vinod/projects_1/Heart_stroke_ANN_/Data/healthcare-dataset-stroke-data.csv"

if __name__ == "__main__":
    try:
        output = strat_batch_prediction(input_file_path=file_path)

    except Exception as e:
        print(e)