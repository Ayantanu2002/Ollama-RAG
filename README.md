# Ollama-RAG

Ollama-RAG is a project that demonstrates the use of the **Ollama** model in different types of chat systems and frameworks. The project is organized into three main parts:

1. **Basic Chat System** - A simple chat interface using the Ollama model.
2. **Multimodal Chat System** - A chat system that handles both text and image inputs using Ollama.
3. **Retrieval-Augmented Generation (RAG) System** - A system that answers queries using web data, specifically Wikipedia articles, in combination with the Ollama model.


## Model Loading with Ollama

Before using any model in your application, you need to load it using the `ollama` CLI. Here's how to do it:

1. **Install Ollama**:  
   Follow the official instructions to install Ollama for your system.  
   - [Ollama Installation Guide](https://ollama.com)

2. **Run the model**:  
   After you install Ollama, open your terminal and load a model by running the following command:

```bash
ollama run <model_name>


