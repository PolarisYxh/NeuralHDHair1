运行docker
docker run --gpus all --name hair-server --shm-size 100G -d  -p 50086:50086 -w /app -v "$(pwd):/app" hair-base    bash /app/service/run_services.sh

docker run --gpus all --name hairtrain --shm-size 100G -d -w /app -v $(pwd):/app/NeuralHDHair -v /nvme0/yangxinhang/HairStrand:/app/HairStrand -it hairstrand-base   /bin/bash

export PYTHONPATH=~/NeuralHDHair/Code:~/NeuralHDHair:$PYTHONPATH
python service/service.py

service/config.json里面修改设置
Code/Models/Local_filter.py 里面    15,16行修改设置    
use_add_info = True
use_ori = False