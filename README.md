# Large Language Model Chunkizer
In this repository we use Azure OpenAI GPT-4o to split a document in a consistent way based on the concept of "idea"

When splitting blocks into chunks, we tackle the potential problem of adjacent paragraphs that express the same idea but are initially separated into distinct blocks.

This is important because splitting related information can lead to a loss of context and negatively affect the understanding of the content when processed by the model.

To mitigate this, we take the last two chunks (if they exist) derived from the current block and append them to the beginning of the next block before analyzing it.

This ensures that related concepts are kept together, preserving their context and improving the overall coherence of the information.

This process is repeated for all remaining blocks.

See an example  [LLMChunkizer.ipynb](LLMChunkizer.ipynb)

See the library [LLMChunkizerLib](LLMChunkizerLib/)