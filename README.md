\# Keyword Spotting on ESP32 (TinyML)



Recognises 10 spoken keywords (yes, no, up, down, left, right, on, off, stop, go)

on an ESP32 using a small int8 CNN and TensorFlow Lite Micro.

Audio is captured on the laptop and streamed to the ESP32 over USB.



Based on: https://github.com/priyanshscpp/TFLite-Micro-Keyword-Spotting-on-ESP32-Arduino



\## How to run

1\. cd python; pip install -r requirements.txt

2\. python 01\_prepare\_dataset.py

3\. python 02\_train\_export\_tflite.py

4\. python 03\_export\_headers.py

5\. Flash firmware/kws\_esp32/kws\_esp32.ino (Arduino-ESP32 core 2.0.17, library Chirale\_TensorFlowLite)

6\. python tools/stream\_to\_esp32.py --port COM5 --in-dev 2 --gain 3



Results: float test accuracy 90.3%, noisy test accuracy 83.3%.

int8   test\_adc acc: 83.00%

