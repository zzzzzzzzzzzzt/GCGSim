date=`date +%Y-%m-%d`
train_name=CPRGsim_AIDS700nef
mkdir -p logs/$date/

nohup python main.py \
--dataset AIDS700nef \
--gpu_id 0 \
--train_name ${train_name} \
>logs/$date/${train_name} 2>&1 &