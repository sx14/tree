#! /bin/bash
echo "Preparing virtual environment ..."
virtualenv -p python2.7 env
echo "Preparing virtual environment ... success!"
source env/bin/activate
echo "Activate virtual environment ... success!"
echo "Installing requirements for TreeMeasure-python-part (this may take a long time) ..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "Installing requirements for TreeMeasure-python-part ... success!"