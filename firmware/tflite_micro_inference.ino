// tflite_micro_inference.ino (skeleton)
// - include model_data.h produced by: xxd -i model.tflite > model_data.h
// - add TensorFlowLite Micro library to Arduino
#include "model_data.h"   // contains g_model_data array
#include <TensorFlowLite.h> // (pseudo include — depends on library)
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;
const int PRESSURE_PIN_1 = A0;
const int PRESSURE_PIN_2 = A1;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  if (!mpu.begin()) {
    Serial.println("MPU begin fail");
    while(1) delay(10);
  }
  // TFLite Micro setup:
  // - create arena buffer
  // - create model = tflite::GetModel(g_model_data)
  // - build interpreter, allocate tensors
  // - obtain input/output pointers/indices
  Serial.println("TFLite Micro ready (skeleton)");
}

void loop() {
  // Collect a window of sensor samples (WINDOW length)
  // Preprocess (scale) exactly as during training
  // Fill input tensor with quantized values (int8) or float depending on model
  // interpreter->Invoke()
  // read output tensor, map to labels
  // print result on Serial or LED
}
