# FastCqt

The [constant-Q transform feature(CQT)](https://en.wikipedia.org/wiki/Constant-Q_transform), is often useful for some recognized task, 
especiall for some task about music.
This repo is for c++ code for Efficient solver of CQT and also offer pyWrapper for easily using.

## How to use it
```bash
git clone https://github.com/Liu-Feng-deeplearning/FastCqt.git 
cd FastCqt
```
### prepare for eigen and fftw
```bash
cd 3rd
tar -zxf eigen-3.3.7.tar.gz  # eigen do not need to install

tar -zxf fftw-3.3.10.tar.gz
./configure --enable-shared --enable-float  -prefix=${PWD}/build 
make CFLAGS="-fPIC"
make install

cd ../
``` 
Now we prepared 3rd for FastCqt

### build
```bash
mkdir build && cd build && cmake ../
make -j
```
After building, we get libFastCqt.so

### usage 

Besides of using c++ code, we also offer py-wrapper for the cqt feature. 
Example python code can be seen at src/PyWrapper.py 

```python
import librosa, FastCqt
import numpy as np
sig, _ = librosa.load("demo/demo.wav", sr=16000)
sig = np.ascontiguousarray(sig, dtype=np.float32)
cpp_cqt_ext = FastCqt(sample_rate=16000, hop_size=0.04)
cpp_cqt = cpp_cqt_ext.compute_cqt(sig)
print("input length:{}".format(np.shape(sig)))
print("output shape:{}".format(np.shape(cpp_cqt)))
```

we can get result as below: 
```text
input length:(1459200,)
output shape:(2280, 96)
```

### Toy Example

Compared with other speech features such as mel, the biggest advantage of the Cqt 
is that it is very convenient to handle the shift of the pitch. Here is an example of audio "Red high-heeled shoes"(红色高跟鞋 in Chinese) by Yaxun Xiao. 
We cut of signal with head 30s and compute cqt featrue with hop-size of 0.04, so we get matrix of 96x750(or 750x96).
Compare init data and pitch-shifted data(increase by 20%), we can see than time domain pitch-shift equals to 
frequence domain role alone with freq-axis. And it is useful at some task because we can do data augmentation easily.
 
<div style="text-align: center"><img src="cqt_pitch.png?raw=true" width="800" /></div>


### Efficiency 

This implement is only an **approximation** algorithm which is to multiply between fft 
of signal and transformed matrix. It is fast enough and result is verified by some  
different task such as Cover-song-identification(CSI). According to my experience, 
set hop_size as 0.04 is suitable for audio with sample-rate of 16k. 

This is benchmark for FastCqt and librosa.cqt:

chunks[s]| FastCqt rtf | librosa.cqt rtf
:---:|:---:|:---:|
5|0.0051|0.0107
15|0.0054|0.0078
45|0.0054|0.0069
135|0.0037|0.0045

Note: when program is run, we use "task -c 1" to make sure cpu usage is about 100%

