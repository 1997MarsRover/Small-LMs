## Something about Tokenization

*from @AndrejKarpathy*

Tokenization is at the heart of much weirdness of LLMs. Do not brush it off.

    Why can't LLM spell words? Tokenization.
    
    Why can't LLM do super simple string processing tasks like reversing a string? Tokenization.
    
    Why is LLM worse at non-English languages (e.g. Japanese)? Tokenization.
    
    Why is LLM bad at simple arithmetic? Tokenization.
    
    Why did GPT-2 have more than necessary trouble coding in Python? Tokenization.
    
    Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? Tokenization.
    
    What is this weird warning I get about a "trailing whitespace"? Tokenization.
    
    Why the LLM break if I ask it about "SolidGoldMagikarp"? Tokenization.
    
    Why should I prefer to use YAML over JSON with LLMs? Tokenization.
    
    Why is LLM not actually end-to-end language modeling? Tokenization.
    
    What is the real root of suffering? Tokenization.

Huggingface provides two versions of tokenizers libraries:

 - Full python implementation of the tokenizer
 
 - Rust Implementation, [tokenizers](https://github.com/huggingface/tokenizers), with a focus on performance and versatility. Take a [tour](https://huggingface.co/docs/tokenizers/quicktour). Its an amazing library

 The Rust

What are the different algorithms one can use?✅
How do you make a tokenizer...

