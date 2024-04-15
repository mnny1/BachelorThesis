import pandas as pd
import json


def read_dt(path):
    df = pd.read_csv(path)
    df['file_name'] = df['filename'].apply(lambda x: x.split('/')[-1] + '.jpeg')
    df['image_id'] = df['filename'].factorize()[0]
    df['image_id'] += 1
    return df


def create_json(df):
    tmp = df[["category_id", "label_id"]].drop_duplicates()
    tmp = tmp.rename(columns={"category_id": "id", "label_id": "name"})
    categories_list = tmp[['id', 'name']].to_dict(orient='records')

    tmp = df[["image_id", "width", "height", "file_name"]].drop_duplicates()
    tmp = tmp.rename(columns={"image_id": "id"})
    images_list = tmp[['id', 'width', 'height', 'file_name']].to_dict(orient='records')

    annotations_list = df.apply(lambda row: {
    'id': row['bbox_id'],
    'image_id': row['image_id'],
    'category_id': row['category_id'],
    'area': (row['xmax']-row['xmin']) * (row['ymax']-row['ymin']),  # inference using COCOEvaluator key error: 'area'
    'bbox': [row['xmin'], row['ymin'], row['xmax']-row['xmin'], row['ymax']-row['ymin']],
    'iscrowd': 0
    }, axis=1).tolist()

    out = {"categories": categories_list, "images": images_list, "annotations": annotations_list}
    return out


def save_json(df, output_file):
    out = create_json(df)
    with open(output_file, 'w') as f:
        json.dump(out, f)


if __name__ == "__main__":

    path = "data/test_set_v1_coco_format/csv/test_set.csv"
    output_file = "data/test_set_v1_coco_format/jsons/test_instances_v1.json"
    
    df = read_dt(path)
    save_json(df, output_file)