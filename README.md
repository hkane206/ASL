## Inspiration
Post-pandemic, the world had become increasingly reliant on video conferencing platforms like Zoom to stay connected and continue business, education, and social interactions. However, one glaring issue persisted – the lack of accessibility for individuals who rely on American Sign Language (ASL) to communicate.

Our team of passionate individuals, each bringing unique skills and experiences to the table, came together with a shared motivation to bridge this accessibility gap. We were driven by a collective desire to empower the Deaf and hard-of-hearing community by making virtual meetings and conversations more _inclusive and accessible_.

## What it does
**Gesture** is a cutting-edge application that integrates with Zoom to provide real-time translation of the American Sign Language (ASL) alphabet into text. 

## How we built it
**Parts of Gesture**
1. Video Capture and Processing
2. ASL Alphabet Dataset Training
3. Processing into Sentences
4. Zoom Integration
### Video Capture and Processing
We employed pyvirtualcam to serve as a virtual overlay camera within Zoom. Utilising OpenCV, the application identifies hand movements and captures a static image at regular intervals, whenever it detects a user's hand in view. For the image processing, we used MediaPipe to perform precise hand landmarking, generating nodes at crucial joints and key points on the hands.
### ASL Alphabet Dataset Training
We obtained out training data from the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download), and employed MediaPipe to identify hand landmarks within static images. Subsequently, we trained a machine learning model using TensorFlow that identifies the ASL signs and translates it into English letters.
### Processing into Sentences
We assemble the individual letters into words, which are then processed through a spell checker. Following this, the refined words are fed into an NLP model to correct and improve grammatical structure.
                                
**Example**

    hi Hack MIT! Tis is Gesture, an ASL trnzlation ap. We aree sO exsited to b here!
**Result**  
                      
    Hi Hack MIT! This is Gesture, an ASL translation app. We are so excited to be here!

### Zoom Integration
For bidirectional communication, we utilise zoom_cc for speech to text (closed captions). Futhermore, the virtual camera video feed is integrated into Zoom.

## Challenges we ran into
Initially, our intention was to utilise a comprehensive dataset containing more than 2,000 ASL gestures. However, after refining the dataset repository, we realised that our hardware lacked the robust GPUs needed for the computational demands of training and testing the data.

By this time, it was already late in the evening. We opted to divide into two sub-teams to concurrently work on two different projects: Gesture, and an EEG system designed to monitor the progression of Alzheimer's disease. Ultimately, we chose to proceed with Gesture, believing it offered the greatest potential for rapid development and meaningful community impact.

## Accomplishments that we're proud of
We're proud of being able to not sleep to finish two concurrent projects.

## What we learnt
We learnt that 24 hours is not enough.

## What's next for Gesture
Use ASL Gesture to text and speech translation.
