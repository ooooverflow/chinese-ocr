conda create -n chineseocr python=3.6
pip install numpy scipy matplotlib pillow
pip install easydict opencv-python torch h5py PyYAML
pip install cython


# for gpu
pip install tensorflow-gpu
chmod +x ./ctpn/lib/utils/make.sh
cd ./ctpn/lib/utils/ && ./make.sh

# for cpu
# pip install tensorflow==1.3.0
# chmod +x ./ctpn/lib/utils/make_cpu.sh
# cd ./ctpn/lib/utils/ && ./make_cpu.sh