export SG_ROOT=data/ShareGPT4V/data

echo 'starting download for llava images'
mkdir -p $SG_ROOT/llava/llava_pretrain
wget -O $SG_ROOT/images.zip \
     https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
unzip -q $SG_ROOT/images.zip -d $SG_ROOT/llava/llava_pretrain/
echo 'finished downloading llava images'

echo 'starting download for coco images'
mkdir -p $SG_ROOT/coco
wget -O $SG_ROOT/train2017.zip \
     http://images.cocodataset.org/zips/train2017.zip
unzip -q $SG_ROOT/train2017.zip -d $SG_ROOT/coco/
echo 'finished downloading coco images'
