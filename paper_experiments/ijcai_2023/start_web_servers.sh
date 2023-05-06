#!/bin/bash
python3 -m ReadEm.serve -b 0.0.0.0 &
jupyter notebook --port 9990 --no-browser --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''