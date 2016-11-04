# CommonVariableLearning

This repository contains tensorflow code for the paper "Learning by Coincidence: Siamese Networksand Common Variable Learning" by Uri Shaham and Roy Lederman.

The script Siamese_commonDemoRunner.py creates a dummy dataset where data is measured through two sensors, which measure common variable x and sensor-specific variables y,z.
The script trains a Siamese network that preserves the common variability and discards the sensor-specific variability.

Once the net is trained, we use a diffusion map  and singular value decomposition to obtain a low dimensional embedding of the data from each sensor.