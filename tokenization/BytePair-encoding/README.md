## Byte-Pair encoding

**What is byte-pair encoding?**

Byte-Pair Encoding is a method for compressing text data by representing words and phrases as a sequence of bytes. It's based on the idea of replacing frequent word pairs with a single byte, and then recursively applying this process to the resulting bytes. Take a look on an article of it on [wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding)

**How does BPE work?**

  *Frequency analysis*: The algorithm analyzes the frequency of word pairs in the text data.  

  *Pair selection*: The most frequent word pairs are selected and replaced with a single byte, called a "token".

  *Tokenization*: The text is tokenized into individual words or subwords.

  *Encoding*: Each token is encoded as a sequence of bytes.

  *Recursion*: Steps 2-4 are repeated recursively until the desired level of compression is reached.


