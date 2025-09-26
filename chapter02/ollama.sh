#!/bin/bash

name=ollama

case "$1" in
  down)
    docker stop $name
  ;;

  run)
    model=$2
    if [[ -z "$model" ]]; then
      echo "用法：$1 run <模型名称>"
      exit 2
    fi

    docker exec -it ollama ollama run $2
  ;;

  up)
    docker run -td --rm -v $PWD/_ollama:/root/.ollama --name $name ollama/ollama:0.11.4 serve

    ip=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $name`
    echo "ollama 的监听地址为 $ip:11434"
  ;;

  *)
    echo "未知命令 '$1'"
    exit 1
  ;;
esac

