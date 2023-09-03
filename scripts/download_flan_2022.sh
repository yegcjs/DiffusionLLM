mkdir -p data/flan_2022
cp scripts/flan_2022_ratio.json data/flan_2022/ratio.json
cd data/flan_2022

wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/cot_fs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/cot_fs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/cot_zs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/cot_zs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/dialog_fs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/dialog_fs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/dialog_zs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/dialog_zs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/flan_fs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/flan_fs_opt_train_part1.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/flan_fs_opt_train_part2.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/flan_fs_opt_train_part3.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/flan_zs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/flan_zs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_fs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_fs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_zs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_zs_opt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_fs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_zs_noopt_train.jsonl.gz
wget https://huggingface.co/datasets/SirNeural/flan_v2/resolve/main/niv2_zs_opt_train.jsonl.gz

unpigz *.gz

cd ../..
