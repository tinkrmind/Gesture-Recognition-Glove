# Gesture recognition with hardware neural networks on the Intel Curie

The Arduino 101 board uses the Intel Curie chip. This chip is meant for wearables and contains an accelerometer, an M0 processor, bluetooth stack, RTC and importantly for this project a hardware Pattern Matching Engine(PME). For this project I will use the Arduino IDE and CuriePME library from Intel.

The CuriePME takes in 128 byte vectors as input and classifies them into one of up to 128 classes. The model is trained by feeding in multiple vectors for a given class. Since the data acquisition and training both happen on the same chip this is called EDGE computing.

Here's a video of it in action:

[![Gesture Recognition](https://i.imgur.com/YFwuPzO.jpg)](https://vimeo.com/301956590)
