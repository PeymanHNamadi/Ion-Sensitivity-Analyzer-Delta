# assume that the container already has activated the right environment
# asumme current working directory is at the top level of this repo
panel serve Sensitivity_IonStudy.py --address 0.0.0.0 --port 80 --allow-websocket-origin="*"
