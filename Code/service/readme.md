运行docker
docker run --gpus all --name segmentall-server -d  -p 50083:50083 -w /app -v "$(pwd):/app" segmentall-base    bash /app/run_services.sh