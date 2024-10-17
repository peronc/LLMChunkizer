from langchain_openai.chat_models.azure import AzureChatOpenAI
import tiktoken

class llm_chunkizer:
    @staticmethod
    # Function to estimate token count for a text block
    def estimate_token_count(text: str, tokenizer_model: str = "gpt-4"):
        """
        Given a text we estimate token need
        Args:
            text (str): text for which to estimate needed tokens 
            tokenizer_model (str, optional): tokenizer to use for estimation . Defaults to "gpt-4".
        """
        encoding = tiktoken.encoding_for_model(tokenizer_model)
        return len(encoding.encode(text))

    @staticmethod
    # Function to split document into large blocks of a specific token size
    def split_document_into_blocks(paragraphs, block_token_limit: int = 5000):
        """
        The main idea is to extract text from a document as paragraphs or sections from the original content. 
        Then, we create a list of text blocks, each formatted to fit within the token limit allowed 
        by the LLM for prompts (in this case, defined by block_token_limit).
        Args:
            paragraphs: an array of strings that represent the paragraph of a document
            block_token_limit (int, optional): max block size in token. Defaults to 5000.
        """
        blocks = []
        current_block = ""
        current_token_count = 0

        for paragraph in paragraphs:
            paragraph_token_count = llm_chunkizer.estimate_token_count(paragraph) 
            # If adding this paragraph exceeds the token limit, finalize the current block
            if current_token_count + paragraph_token_count > block_token_limit:
                blocks.append(current_block.strip())
                # Start new block with current paragraph
                current_block = paragraph  
                current_token_count = paragraph_token_count
            else:
                current_block += paragraph.strip() + "\n"
                current_token_count += paragraph_token_count

        # Add any remaining text as the final block
        if current_block:
            blocks.append(current_block.strip())

        return blocks

    @staticmethod
    # Function to chunk text using LLM
    def chunk_text_with_llm(llm: AzureChatOpenAI, blocks):
        """
            In this function we take an array with blocks of text and ask the LLM to divide them into self-consistent chunks
            using a prompt that ask to keep toghether text that express a concrete idea.
            
            Each block used for input, was created by putting together paragraphs extracted from a text document. 
            When grouping into blocks, we tried to respect the maximum token length foreseen by the prompt
            without considering that the content at the end of one block and the beginning of the next 
            could be be kept together because they deal with a single topic.
            
            We address the potential issue of adjacent paragraphs that convey the same idea being split into separate blocks.
            This is important because splitting related information can lead to a loss of context and negatively affect
            the understanding of the content when processed by the model.
            To mitigate this, we take the last two chunks (if they exist) from the current block and append them 
            to the beginning of the next block before analyzing it.
            This ensures that related concepts are kept together, preserving their context and improving the overall
            coherence of the information.
            This process is repeated for all remaining blocks.
        Args:
            llm (AzureChatOpenAI): Azure OpenAI model
            blocks (array: str): a list of block to split in consistent chunks 

        Returns:
            array: str: an array of chunk splitted by idea
        """
        final_chunks = []
        last_chunk = ""
        last_chunk_2 = ""
        last_chunk_1 = ""

        for block in blocks:
            text = last_chunk + "\n" + block
            prompt = [
                {"role": "system", "content": "You are an assistant that helps divide documents into logical chunks based on complete ideas."},
                {"role": "user", "content": f"Please split the following text into logical chunks, using '!-!-!-!-!-!-!-!-!-!-!' to separate them. \n\n{text}"}
            ]
            #invoke llm with prompt
            response = llm.invoke(prompt)        
            text_to_split = response.content
            #split text by '!-!-!-!-!-!-!-!-!-!-!'. Each element of the splitted_array is an autoconsistent chunk.
            splitted_array = text_to_split.split('!-!-!-!-!-!-!-!-!-!-!')
            
            #retain last 2 chunk for the this block and use them to the next
            last_chunk_1 = splitted_array.pop() if splitted_array else ""
            last_chunk_2 = splitted_array.pop() if splitted_array else ""          
            last_chunk = last_chunk_2 + "\n" + last_chunk_1
            
            final_chunks.extend(splitted_array)
                
        final_chunks.append(last_chunk_2)
        final_chunks.append(last_chunk_1)
        return final_chunks
