# ZED Face privacy SVO Recording

Record SVO while displaying people face blurred using ZED SDK head detector

```
Usage: ./ZED_face_privacy <output.svo> [output_file.avi] [--flip]
```

Note: The SVO only record raw stereo video feed (YUV Left and Right unrectified images), the face blur is only applied to a temporary image and the output avi file, not to the SVO data.
