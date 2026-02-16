**Multimodal Emotion Recognition using Speech and Text**

****Applicant:**** Hariprasad Chinimilli

****Target:**** Assignment 2 - Intern/Research Assistant Position, LTRC, IIIT Hyderabad

****Project Overview****
In this project, I built a system to recognize emotions from two different inputs: **Speech (Audio)** and **Text (Transcripts)**. The goal was to see how each modality performs individually and whether combining them (Fusion) improves the overall emotion detection. I used the **Toronto Emotional Speech Set (TESS)** dataset, which contains 2,800 high-quality audio samples covering 7 cardinal emotions.

****Technical Approach & Architecture****
I followed a modular pipeline approach as per the assignment requirements:

1.**Speech Pipeline (Temporal Modelling):** I used **40 MFCC features** for extraction and an **LSTM (Long Short-Term Memory)** network. LSTMs are perfect here because they can learn emotional patterns over time in an audio signal.

2.**Text Pipeline (Contextual Modelling):** I used a pre-trained **BERT-base-uncased** model. Since the text phrases were short carrier sentences, BERT helped in capturing the contextual meaning of the target words.

3.**Fusion Pipeline (Multimodal):** I implemented **Late Fusion** by concatenating the representations from both the LSTM (Speech) and BERT (Text) branches into a single unified classifier.

****Experimental Results****


<img width="327" height="121" alt="image" src="https://github.com/user-attachments/assets/cdd883f7-db32-4db8-a014-3747d3c9417a" />


****Key Observations & Analysis****
**Speech Performance:** The Speech model achieved nearly perfect accuracy (99.46%) because the TESS dataset has very distinct acoustic cues (pitch and energy) for each emotion.

**The Text Challenge:** Text accuracy was significantly lower (13.75%) because the "carrier phrase" in the dataset ("Say the word...") is identical for all emotions, providing very little semantic difference for the BERT model to learn.

**Separability:** The t-SNE visualization confirms that the Temporal Modelling block successfully clustered emotions like 'Angry' and 'Sad' into distinct groups based on their audio features.
