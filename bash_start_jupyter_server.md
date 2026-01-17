jupyter notebook \
  --ServerApp.allow_origin="https://colab.research.google.com" \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --no-browser \
  --MappingKernelManager.cull_idle_timeout=86400 \
  --ServerApp.websocket_ping_interval=0
