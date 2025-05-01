# Hugging Face Tutorial

This repository contains practical examples of using Hugging Face's libraries for various Natural Language Processing (NLP) and Machine Learning tasks.

## Installation

Install the necessary packages:

```bash
pip install transformers datasets evaluate accelerate sentencepiece
```

## Examples

### 1. Text Classification

Using a pre-trained model for sentiment analysis:

```python
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze some text
texts = [
    "I love using Hugging Face's transformers library!",
    "This movie was terrible and boring.",
    "The food at the restaurant was decent, but the service was slow."
]

results = sentiment_analyzer(texts)

# Display results
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['score']:.4f}\n")
```

### 2. Named Entity Recognition

Identifying entities like people, organizations, locations in text:

```python
from transformers import pipeline

# Initialize the NER pipeline
ner_pipeline = pipeline("ner")

# Text to analyze
text = "My name is Sarah and I work at Microsoft in Seattle. I met with John from Google last week in New York."

# Get entities
entities = ner_pipeline(text)

# Group entities by word
grouped_entities = {}
for entity in entities:
    word = entity["word"]
    if word not in grouped_entities:
        grouped_entities[word] = []
    grouped_entities[word].append(entity)

# Display entities nicely
for word, word_entities in grouped_entities.items():
    entity_type = word_entities[0]["entity"]
    score = word_entities[0]["score"]
    print(f"Word: {word}\nEntity Type: {entity_type}\nConfidence: {score:.4f}\n")
```

### 3. Text Generation

Generating text completions with GPT-2:

```python
from transformers import pipeline

# Initialize text generation pipeline with GPT-2
generator = pipeline('text-generation', model='gpt2')

# Generate text from prompts
prompts = [
    "Once upon a time,",
    "The future of artificial intelligence",
    "Hugging Face is"
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    result = generator(prompt, max_length=50, num_return_sequences=1)
    print(f"Generated: {result[0]['generated_text']}\n")
```

### 4. Question Answering

Using a model to answer questions based on provided context:

```python
from transformers import pipeline

# Initialize the question answering pipeline
qa_pipeline = pipeline("question-answering")

# Context and questions
context = """
Hugging Face is an AI community and platform that was founded in 2016 by Clément Delangue, 
Julien Chaumond, and Thomas Wolf. The company is based in New York City, USA and Paris, France. 
Hugging Face is known for its transformers library which provides pretrained models for natural 
language processing tasks. In 2021, the company raised $40 million in Series B funding to 
expand its team and services.
"""

questions = [
    "When was Hugging Face founded?",
    "Who founded Hugging Face?",
    "What is Hugging Face known for?",
    "How much funding did Hugging Face raise in Series B?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}\n")
```

### 5. Image Classification

Classifying images using a pre-trained vision model:

```python
from transformers import pipeline

# Initialize the image classification pipeline
image_classifier = pipeline('image-classification')

# Path to your image
image_path = "path/to/your/image.jpg"

# Classify the image
result = image_classifier(image_path)

# Display the top predicted classes
for prediction in result:
    print(f"Class: {prediction['label']}")
    print(f"Confidence: {prediction['score']:.4f}\n")
```

### 6. Translation

Translating text between languages:

```python
from transformers import pipeline

# Initialize translation pipelines
en_to_fr = pipeline("translation_en_to_fr")
en_to_de = pipeline("translation_en_to_de")
en_to_ro = pipeline("translation_en_to_ro")

# Text to translate
text = "Hugging Face is a technology company that develops tools for building applications using machine learning."

# Translate to different languages
print(f"Original: {text}")
print(f"French: {en_to_fr(text)[0]['translation_text']}")
print(f"German: {en_to_de(text)[0]['translation_text']}")
print(f"Romanian: {en_to_ro(text)[0]['translation_text']}")
```

### 7. Summarization

Summarizing long texts:

```python
from transformers import pipeline

# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Long text to summarize
long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence
displayed by humans or animals. Leading AI textbooks define the field as the study of "intelligent agents": 
any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions 
that humans associate with the human mind, such as "learning" and "problem solving", however this definition is 
rejected by major AI researchers.

AI applications include advanced web search engines, recommendation systems, 
human speech recognition, autonomous driving, automated decision-making and competing at the highest level in strategic 
game systems. As machines become increasingly capable, tasks considered to require "intelligence" are often removed 
from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is 
frequently excluded from things considered to be AI, having become a routine technology.

The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be
made to simulate it". This raises philosophical arguments about the mind and the ethics of creating artificial beings
endowed with human-like intelligence. These issues have been explored by myth, fiction and philosophy since antiquity.
"""

# Generate a summary
summary = summarizer(long_text, max_length=150, min_length=50, do_sample=False)

print(f"Summary: {summary[0]['summary_text']}")
```

### 8. Speech Recognition

Transcribing speech to text:

```python
from transformers import pipeline
from datasets import load_dataset
import IPython.display as ipd

# Initialize speech recognition pipeline
transcriber = pipeline("automatic-speech-recognition")

# Load a sample audio file from the Hugging Face datasets library
sample_audio = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="validation[:3]")

# Display and transcribe each audio sample
for i, audio_sample in enumerate(sample_audio):
    print(f"\nAudio sample {i+1}:")
    
    # Play the audio (will work in Jupyter Notebook)
    ipd.display(ipd.Audio(audio_sample["audio"]["array"], rate=audio_sample["audio"]["sampling_rate"]))
    
    # Transcribe the audio
    transcription = transcriber(audio_sample["audio"])
    
    print(f"Transcription: {transcription['text']}")
    print(f"Reference: {audio_sample['sentence']}") 
```

## Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers GitHub Repository](https://github.com/huggingface/transformers)
- [Hugging Face Model Hub](https://huggingface.co/models)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
