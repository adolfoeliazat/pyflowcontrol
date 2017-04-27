# pyflowcontrol
Using Raspberry, OpenCV, python and PIR sensor to do flow control of persons and cars.
## Objective
This project intent is to use a Raspberry Pi 3 as a central system to process and manage flow of people and cars in buildings or condominiums.
## Methodology
1. Moving object detection via PIR sensor
2. Wait for QR code
3. If detects a QR code, retrieve information (LBP histogram) about this person's QR code in the database
4. Compare retrieved data with real time data using some kind of classificator
5. If it matches, then allow entrance
