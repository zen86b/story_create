import re


def split_into_chunks(text):
    # Split the text into sentences
    sentences = re.findall(r'[^.!?;]+[.!?;]', text)

    # Initialize an empty list to store the chunks
    chunks = []

    # Iterate through the sentences
    for i in range(0,len(sentences),2):
        # If there are enough sentences remaining to form a two-sentence chunk
        if i + 1 < len(sentences):
            # Create a chunk from the current sentence and the next sentence
            chunk = sentences[i] + sentences[i + 1]
            chunks.append(chunk.strip())
        else:
            chunk = sentences[i]
            chunks.append(chunk.strip())

    return chunks

if __name__ == "__main__":
    with open("tmp.txt", "r") as f:
        content = f.read()

        s = split_into_chunks(content)
        for i in s:
            print("-"*50)
            print(i)
