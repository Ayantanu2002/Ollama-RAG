import ollama

with open('image.jpg', 'rb') as f:
    image = f.read()
    
    response = ollama.chat(
        model='llava',
        messages=[
            {"role": "user", "content": "What is this image about?", "image": [image],},
        ],
        
    )
    
print(response['message']['content'])