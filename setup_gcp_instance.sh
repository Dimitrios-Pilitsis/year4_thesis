#Install CUDA
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
sudo nvidia-smi
#Install pip
sudo apt install wget -y
wget "https://bootstrap.pypa.io/get-pip.py"
sudo apt install python3-distutils -y
python3 get-pip.py
export PATH="$HOME/.local/bin:$PATH"
sudo apt install git -y
#Clone repository
git clone https://github.com/Dimitrios-Pilitsis/year4_thesis.git
cd year4_thesis
#Install all dependencies
pip3 install -r requirements_gpu.txt
