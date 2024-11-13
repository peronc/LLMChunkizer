# Large Language Model Chunkizer
In this repository we use Azure OpenAI GPT-4o to split a document in a consistent way based on the concept of "idea"

When splitting blocks into chunks, we tackle the potential problem of adjacent paragraphs that express the same idea but are initially separated into distinct blocks.

This is important because splitting related information can lead to a loss of context and negatively affect the understanding of the content when processed by the model.

To mitigate this, we take the last two chunks (if they exist) derived from the current block and append them to the beginning of the next block before analyzing it.

This ensures that related concepts are kept together, preserving their context and improving the overall coherence of the information.

This process is repeated for all remaining blocks.

See an example  [LLMChunkizer.ipynb](LLMChunkizer.ipynb)

See the library [LLMChunkizerLib](LLMChunkizerLib/)

---

## Article on Towards Data Science
In my latest article published on Towards Data Science, I explore how large language models (LLMs) can revolutionize the way we segment and analyze documents. 

This technique, known as "document chunking," is essential for:
- Enhancing information retrieval accuracy,
- Managing large documents more efficiently,
- Unlocking knowledge from fragmented blocks of text.

If you're interested in learning how to apply these techniques to optimize document processing, check out my full article on Towards Data Science here: [Efficient Document Chunking Using LLMs: Unlocking Knowledge One Block at a Time](https://medium.com/@peronc79/355717a88c5c?sk=1cc4e46c40708d5057d54da391035cfa) 🚀

Here you can find a pdf copy of the [article](docs/WEB_Article_Efficient_Document_Chunking_Using_LLMs_Unlocking_Knowledge_One_Block_at_a_Time_by_Carlo_Peron_Oct_2024_TowardsDataScience.pdf)
