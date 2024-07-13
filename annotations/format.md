This annotation json is from "PPDM: Parallel Point Detection and Matching for Real-time Human-Object Interaction Detection". (CVPR2020)
github repo: https://github.com/YueLiao/PPDM

base_url: /home/gkf/HOI_datasets/hico_20160224_det/images/train2015

The content of "trainval_hico.json" is a list, each item is a dict as:
```
{
    'file_name': 'HICO_train2015_00000001.jpg', 
    'img_id': 1, #  img_id == int(filename[-12:-4]), you can assert this
    'annotations': [
        {'bbox': [207, 32, 426, 299], 'category_id': 1}, # box format: xyxy; # category_id in COCO format, not continues (and `1` is human).
        {'bbox': [58, 97, 571, 404], 'category_id': 4}
    ], 
    'hoi_annotation': [
        {'subject_id': 0, 'object_id': 1, 'category_id': 73, 'hoi_category_id': 153}, # category_id range: 1~117; hoi_category_id range: 1~600
        {'subject_id': 0, 'object_id': 1, 'category_id': 77, 'hoi_category_id': 154}, 
        {'subject_id': 0, 'object_id': 1, 'category_id': 88, 'hoi_category_id': 155}, 
        {'subject_id': 0, 'object_id': 1, 'category_id': 99, 'hoi_category_id': 156}
    ]
}
```

The content of "test_hico.json" is a list, each item is a dict as:
```
{
    'file_name': 'HICO_test2015_00000017.jpg', 
    'annotations': [
        {'bbox': [2, 2, 639, 475], 'category_id': 1},  # box format: xyxy; # category_id in COCO format, not continues (and `1` is human).
        {'bbox': [392, 128, 633, 340], 'category_id': 40}, 
        {'bbox': [1, 1, 637, 478], 'category_id': 1}, 
        {'bbox': [394, 134, 632, 342], 'category_id': 40}
    ],
    'hoi_annotation': [
        {'subject_id': 0, 'object_id': 1, 'category_id': 37}, 
        {'subject_id': 2, 'object_id': 3, 'category_id': 115}
    ]
}
```