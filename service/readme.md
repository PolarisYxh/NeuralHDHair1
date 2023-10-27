运行docker
docker run --gpus all --name hair-server -d  -p 50086:50086 -w /app -v "$(pwd):/app" hair-base    bash /app/service/run_services.sh

export PYTHONPATH=~/NeuralHDHair/Code:~/NeuralHDHair:$PYTHONPATH
python service/service.py

service/config.json里面修改设置
Code/Models/Local_filter.py 里面    15,16行修改设置    
use_add_info = True
use_ori = False