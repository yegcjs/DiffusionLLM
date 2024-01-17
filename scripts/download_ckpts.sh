HF_ENDPOINT="https://hugggingface.co"

# BASE
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.base.length.tar.gz.part0?download=true -O flan_v2.base.length.tar.gz
tar -xvzf flan_v2.base.length.tar.gz
rm flan_v2.base.length.tar.gz

# LARGE
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.large.length.tar.gz.part0?download=true -O flan_v2.large.length.tar.gz
tar -xvzf flan_v2.large.length.tar.gz
rm flan_v2.large.length.tar.gz

# XL
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xl.length.tar.gz.part0?download=true -O flan_v2.xl.length.tar.gz.part0
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xl.length.tar.gz.part2?download=true -O flan_v2.xl.length.tar.gz.part1
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xl.length.tar.gz.part1?download=true -O flan_v2.xl.length.tar.gz.part2
cat flan_v2.xl.length.tar.gz.part0 flan_v2.xl.length.tar.gz.part1 flan_v2.xl.length.tar.gz.part2 > flan_v2.xl.length.tar.gz
tar -xvzf flan_v2.xl.length.tar.gz
rm flan_v2.xl.length.tar.gz*

# XXL
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xxl.length.tar.gz.part0?download=true -O flan_v2.xxl.length.tar.gz.part0
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xxl.length.tar.gz.part1?download=true -O flan_v2.xxl.length.tar.gz.part1
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xxl.length.tar.gz.part2?download=true -O flan_v2.xxl.length.tar.gz.part2
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xxl.length.tar.gz.part3?download=true -O flan_v2.xxl.length.tar.gz.part3
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xxl.length.tar.gz.part4?download=true -O flan_v2.xxl.length.tar.gz.part4
wget $HF_ENDPOINT/yegcjs/diffusion_llm/resolve/main/flan_v2.base.length/flan_v2.xxl.length.tar.gz.part5?download=true -O flan_v2.xxl.length.tar.gz.part5
cat flan_v2.xxl.length.tar.gz.part0 flan_v2.xxl.length.tar.gz.part1 flan_v2.xxl.length.tar.gz.part2 flan_v2.xxl.length.tar.gz.part3 flan_v2.xxl.length.tar.gz.part4 flan_v2.xxl.length.tar.gz.part5 > flan_v2.xxl.length.tar.gz
tar -xvzf flan_v2.xxl.legnth.tar.gz
rm rm flan_v2.xxl.length.tar.gz*
