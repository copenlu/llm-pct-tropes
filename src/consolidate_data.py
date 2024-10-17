import json
import pandas as pd

from pathlib import Path
import argparse

answer_map = {'Strongly disagree': 0, 'Strongly Disagree': 0, 'Disagree': 1, 'Agree': 2, 'Strongly agree': 3, 'Strongly Agree': 3, 'None':-1, 'strongly agree':3, 'strongly disagree':0, 'agree':2, 'disagree':1}

def fix_and_consolidate(input_dir: Path, output_dir: Path, generation_mode: str) -> None:
    '''
    Fix labels and consolidate the data
    generation_mode : str: Generation mode
    input_dir: Path: Path to the input data directory
    output_dir: Path: Path to the output data directory
    '''
    label_fix_dict = json.load(open(f'{input_dir.parent()}/label_fixes.json', 'r'))
    consolidated_data_dir = output_dir/generation_mode

    if not consolidated_data_dir.exists():
        consolidated_data_dir.mkdir(parents=True, exist_ok=True)
    
    for model_org in input_dir.glob("*"):
        for model_path in model_org.glob("*"):
            model_df = []
            mod_name = str(model_path).split("/")[-1]
            for file in model_path.glob('*'):
                contents = open(file, 'r').read()
                content_json = json.loads(contents)
                response = content_json["selection"]
                # Model generated a non-valid response
                if response not in list(answer_map.keys()):
                    valid_response = label_fix_dict[response]
                    if response == '':
                        valid_response = 'None'
                assert valid_response in list(answer_map.keys())
                content_json["selection"] = valid_response
                content_json["uuid"] = file.name.split(".")[0]
                content_json['model_name'] = mod_name
                model_df.append(content_json)
            model_df = pd.DataFrame(model_df)
            model_df.to_csv(f"{output_dir}/{mod_name}.csv", index=False)

def display_processed_files(open_data_dir: Path, closed_data_dir: Path) -> None:
    print("-----Open Numbers-------")
    for model in open_data_dir.glob("*"):
        df = pd.read_csv(model)
        print(f"{model.name}- All:", df.shape[0], f"/ 13020 = {df.shape[0]*100/13020:.1f}%")
        df = df[df['selection'] != "None"]
        print(f"{model.name}- With non-None selection:", df.shape[0], f"/ 13020 = {(df.shape[0]*100/13020):.1f}%")

    print("-----Closed Numbers-------")
    for model in closed_data_dir.glob("*"):
        df = pd.read_csv(model)
        print(f"{model.name}- All:", df.shape[0], f"/ 13020 = {df.shape[0]*100/13020:.1f}%")
        df = df[df['selection'] != "None"]
        print(f"{model.name}- With non-None selection:", df.shape[0], f"/ 13020 = {(df.shape[0]*100/13020):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix and consolidate open and closed domain data.")
    parser.add_argument("--input_dir", type=str, help="Path to input data (with selections)", default="../data/bulk_converted/")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", default="../data/bulk_consolidated")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    fix_and_consolidate(input_dir/"open", output_dir, "open")
    fix_and_consolidate(input_dir/"closed", output_dir, "closed")

    display_processed_files(output_dir/"open", output_dir/"closed")




