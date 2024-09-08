### How to run the TSR-RDANet model for Gaussian denoising

### 1 dataset download

download the [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), and put it into the /gaussian_denoising/trainsets/.

### 2 get image patches

### 2.1 get color image patches

```
        python utils/utils_image.py
Note: the last few lines of code in this file can be used to modify the data path.   
```    

### 2.2 get grayscale image patches

```
        python RGB_To_Gray.py
```        
   
### 3. Train TSP-RDANet

```
python train.py 

Note: for the training of grayscale and color images, you need to modify the parameters of the gaussian_denoising/options/train_tsp_rdanet.json file, including n_channels, dataroot_H, in_nc and out_nc.
```

### 4. Test TSP-RDANet

```
python test.py
```
