python3 -m venv vlass_venv
cd ~/VLASS_script/non_standard
source vlass_venv/bin/activate
pip install --upgrade pip
pip install flask astropy numpy matplotlib requests pandas

Copy all files and VLASS MSes in a folder then execute following -

python3 VLASS_web_interface.py
If everything goes well then it will print something like this in terminal

Starting VLASS web interface on http://127.0.0.1:5000 (local host with port 5000)

Then open this URL in webbrowser
firefox http://127.0.0.1:5000

To kill, press CTRL+X

For submitting job, it must be open on nmpost server. 
