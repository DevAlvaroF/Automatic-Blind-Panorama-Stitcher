# Automatic-Blind-Panorama-Stitcher
Blind panorama stitcher based on the paper "Automatic Panorama Image Stitching using Invariant Features" by Brown and Lowe

## Highlights
1. Blind stitching. Separates panoramas that do not belong to each other (based on overlapping points transformed to destination place)
2. Images not belonging to each other are not connected in the network and thus excluded (more efficient)
3. Final images are trimmed to reduced black space from geometric transformation

## Drawbacks
- Bundle adjustment takes a long time with many images
- Network theory slows the process considerable when more images from different panoramas are added

## Potential Improvements
- Derivation of Camera Jacobian Matrix 
- Finding the Image in the center that is visually symmetric to the user
- Save images to .png files

## How to run
1. Download the repo
2. Run `main_notebook.ipynb` to visualize the output better o runt the `main.py` file (Image outputs may not visualize correctly on all computers
3. Image folder can be selected in the code section:
```
# --------------------------------------
imageFolder = "images/setBig"
#       images/setBig has 4 Unordered panoramas
#       images/setSmall has 2 Unordered panoramas
#       images/setSingle has 1 panorama
#       images/setSingle2 has 1 panorama
#       images/setSingle3 has 1 panorama
#       images/setSingle4 has 1 panorama
#       images/setSingle5 has 1 panorama
# --------------------------------------
```
4. Configuration of parameters can be changed in `src/config.py`

## Demo Images

### Network Connections
Images not belonging to each other are not connected in the network and thus excluded (saves power)
![image](https://user-images.githubusercontent.com/87340855/218828998-e9516e9a-2eb9-4799-8883-8de8b3e36e01.png)

### Panorama 1 of 4 (Output)
![image](https://user-images.githubusercontent.com/87340855/218829478-a36953d1-d750-4984-87b1-2feb002f7b6f.png)

### Panorama 2 of 4 (Output)
![image](https://user-images.githubusercontent.com/87340855/218829634-01ae91e2-8357-46df-9866-23075e65ad29.png)

### Panorama 3 of 4 (Output)
![image](https://user-images.githubusercontent.com/87340855/218829781-a77d446a-20da-4b66-aec4-d4e9c93a0931.png)

### Panorama 4 of 4 (Output)
![image](https://user-images.githubusercontent.com/87340855/218829891-26f6b3f3-1c0e-43b2-aa8c-33263a16ff72.png)

## References
- https://github.com/freddieb/panoramic-image-stitching
- https://github.com/ppwwyyxx/OpenPano
- https://github.com/preethamam/AutomaticPanoramicImageStitching-AutoPanoStitch
- https://github.com/kluo8128/cs231_project
