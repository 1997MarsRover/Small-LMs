## Something about Tokenization

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

### Some of the algorithms to create a tokenizer

**Simple Tokenization** : This is the most basic approach, where a tokenizer splits the input text into tokens based on whitespace characters (spaces, tabs, newlines, etc.). 

**Regular Expressions (regex)** : This algorithm uses regular expressions to define patterns for tokenization. For example, you can use regex to split on punctuation marks, numbers, or special characters.

**N-gram Tokenization** : This algorithm splits the input text into tokens based on a sliding window of n-grams (sequences of n items). For example, you can use a 2-gram (bigram) to split on word pairs.

**WordPiece Tokenization** : This algorithm is used in the WordPiece algorithm, which is a subword-based tokenization approach. It splits words into subwords (smaller units) and then combines them to form tokens.

**Unigram Tokenization** : This algorithm splits the input text into tokens based on individual words or characters. It's a simple and efficient approach, but may not capture complex linguistic structures.

**Part-of-Speech (POS) Tagging** : This algorithm uses POS tagging to identify the parts of speech (nouns, verbs, adjectives, etc.) and then splits the text into tokens based on these categories.

**Dependency Parsing** : This algorithm uses dependency parsing to analyze the grammatical structure of the input text and then splits it into tokens based on the dependencies between words.

**Hybrid Tokenization** : This algorithm combines multiple tokenization algorithms to create a more robust and accurate tokenization approach.

**Character N-Grams**  : This algorithm splits the input text into tokens based on character n-grams (sequences of n characters).

**Subword Tokenization** : This algorithm splits words into subwords (smaller units) and then combines them to form tokens. This approach is used in the WordPiece algorithm.

**BPE (Byte-Pair Encoding)**  : This algorithm is a variant of the WordPiece algorithm that uses a different encoding scheme to split words into subwords.

**SentencePiece** : This algorithm is a variant of the WordPiece algorithm that uses a different encoding scheme to split words into subwords.

What are the different algorithms one can use?✅
How do you make a tokenizer...

