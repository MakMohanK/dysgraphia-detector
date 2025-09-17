// mpu_pressure_logger.ino
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

const int PRESSURE_PIN_1 = A0;
const int PRESSURE_PIN_2 = A1;

unsigned long sample_interval_ms = 10; // 100 Hz
unsigned long last_sample = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("MPU+Pressure logger starting...");

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip - check wiring!");
    while (1) delay(10);
  }
  // Configure sensor ranges if you want:
  // mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  // mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);

  // Print header (first line)
  Serial.println("timestamp_ms,ax,ay,az,gx,gy,gz,pressure1,pressure2");
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample >= sample_interval_ms) {
    last_sample = now;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp); // fills accel (m/s^2) and gyro (rad/s)

    int p1 = analogRead(PRESSURE_PIN_1);
    int p2 = analogRead(PRESSURE_PIN_2);
    // On RP2040 analogRead returns 0..4095 typically (12-bit)

    // CSV output
    Serial.print(now); Serial.print(",");
    Serial.print(a.acceleration.x); Serial.print(",");
    Serial.print(a.acceleration.y); Serial.print(",");
    Serial.print(a.acceleration.z); Serial.print(",");
    Serial.print(g.gyro.x); Serial.print(",");
    Serial.print(g.gyro.y); Serial.print(",");
    Serial.print(g.gyro.z); Serial.print(",");
    Serial.print(p1); Serial.print(",");
    Serial.println(p2);
  }
  // no delay() necessary — loop driven by timing above
  delay(20); // ~50Hz sampling
}
