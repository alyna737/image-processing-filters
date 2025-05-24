# Interactive Image Processing Web App       

A Flask‑powered web application that lets you upload an image and experiment with classical Digital Image Processing (DIP) techniques—all in real time.  The UI follows a sleek black & grey theme with an accent colour, complete with drag‑and‑drop uploads, sliders, and before/after previews.

##Features

Drag‑and‑drop or click‑to‑upload images

#Filters

Laplacian edge detection

Gaussian blur (custom kernel size)

Median blur (custom kernel size)

Adaptive thresholding with a live slider (keeps colour when using filters)

Morphological operations

Dilation

Erosion

Reset button to revert to the original image anytime

Instant, client‑side previews thanks to Base64 image returns

No deep learning—all effects rely on classical image processing via OpenCV.

#Requirements

Python ≥ 3.10

Flask

OpenCV‑Python

NumPy

Pillow
