# face_recognition

I have used 3 approaches to detect the face in a photo. 

The first is the most basic, using Haar feature-based cascade classifiers to detect faces with a pre-trained model from `opencv`. 
It works very fast, but it immediately loses the face when the head is tilted

The second option is `MTCNN`. It is much slower, but finds faces with excellent accuracy, including key points. 
Further acceleration is possible using gpu

The third option is `MediaPipe` from Google. It works fast and the result is similar to `MTCNN`, even without using GPU. 
So far I have not found any cases where the result is much worse than MTCNN, but so far it seems to be the best solution

## Running

All requirements are listed in the [requirements.txt](requirements.txt) file. 
However, you can simply comment out parts of the code with the other two algorithms, then you won't have to load their corresponding models :) 

Running the application with the `-h` option or without
any arguments yields the following message:

```
usage: main.py [-h] [-o OUTPUT] [--no_show]
               [--algo {haarcascade,mtcnn,mediapipe}]
               input

positional arguments:
  input                 Required. An input to process. The input must be a
                        video file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Optional. Name of the output files to save.
  --no_show             Optional. Don't show output video.
  --algo {haarcascade,mtcnn,mediapipe}
                        Optional. Algorithm for finding faces. Default:
                        haarcascade.
```

You can choose between three facial extraction methods: `haar_cascade`, `mtcnn`, `mediapipe`. 

Using the `--output` option you can specify the name of the output files

If `--no-show` is selected, the current result will not be displayed in a separate window

## Possible improvements. 
1. Find out better what other models are available for this task. 
2. Add gpu for tensorflow, that will drastically improve mtcnn performance and  then compare performance on longer videos.
