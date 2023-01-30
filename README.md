# DVOTR: A Distributed Video Object Tracking framework for Real-time applications
An architecture for distributing object detection tasks from video frames to workers in an edge network.

1. Connect GPU ready workers to session
2. Run admin to view object tracking output
3. Have video file for object tracking task in the correct directory
4. Run the server and begin transmission of video to workers
5. Workers will return inferred bounding boxes to the frame
6. Server will send the ordered frames to admin monitor to draw
7. Server has tolerance for late/out of order frames
8. Workers are chosen for a particular frame based on a preference system
9. Any worker consistantly underdelivering or has issues with connectivity is punished
10. Workers with very poor performance are banned.

Throughput of the network depends entirely on the collaberation of workers, reducing server strain and increasing its bandwidth for other complex tasks.
