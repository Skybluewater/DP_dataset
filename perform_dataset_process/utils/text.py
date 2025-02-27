# video_1, audio_2, text_3, chunk_4
import json
from chonkie import SemanticChunker, SentenceChunker
from .embedding_model import SiliconFlowEmbeddings
from sentence_transformers import SentenceTransformer
from typing import List
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_text_from_audio(audio_path):
    """
    Extracts text from an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Extracted text.
    """
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    from funasr import AutoModel
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                    # spk_model="cam++", spk_model_revision="v2.0.2",
                    )
    res = model.generate(input=audio_path, 
                        batch_size_s=300, 
                        hotword='魔搭')
    with open(audio_path.replace("wav", "json"), "w", encoding="utf-8") as f:
        json.dump(res[0], f, ensure_ascii=False, indent=4)
    return res[0]


def _extract_chunks_from_text(text, **kwargs):
    use_semantic_chunker = kwargs.get("use_semantic_chunker", False)
    if not use_semantic_chunker:
        log.info("Using sentence chunker.")
        print("Using sentence chunker.")
        chunker = SentenceChunker(
            tokenizer_or_token_counter="gpt2",                # Supports string identifiers
            chunk_size=128,                                   # Maximum tokens per chunk
            chunk_overlap=64,                                  # Overlap between chunks
            min_sentences_per_chunk=1,                        # Minimum sentences in each chunk
            min_characters_per_sentence=12,                   # Minimum characters per sentence
            approximate=True,                                 # Use approximate token counting
            delim=[".", "?", "!", "\n", "。", "？", "！"],                      # Delimiters to use for chunking
            include_delim="prev",                             # Include the delimiter in the chunk
            return_type="chunks"                              # Return Chunks or texts only 
        )
        return chunker(text)
    else:
        log.info("Using semantic chunker.")
        print("Using semantic chunker.")
        chunker_semantic = SemanticChunker(
            embedding_model=SiliconFlowEmbeddings(),
            threshold=5,
            chunk_size=64,
            **kwargs
        )
        semantic_chunks = chunker_semantic(text)
        return semantic_chunks


def _kmp_search(pattern, text):
    """
    KMP algorithm for pattern matching.

    Args:
        pattern (str): The pattern to search for.
        text (str): The text to search within.

    Returns:
        list: List of start indices where the pattern is found in the text.
    """
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)
    i = 0
    j = 0
    indices = []
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            indices.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return indices


def align_chunks_with_timestamps(text_path, **kwargs):
    """
    Aligns text chunks with their start and end timestamps.

    Args:
        text_path (str): Path to the text JSON file.
        use_semantic_chunker (bool): Whether to use semantic chunker or not.

    Returns:
        list: List of chunks with their start and end timestamps.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        text_json = json.load(f)
    
    text = text_json["text"]
    meaningful_text = ''.join([char for char in text if char.isalnum() or '\u4e00' <= char <= '\u9fff' or char.isalpha()])
    timestamps = text_json["timestamp"]
    chunks = _extract_chunks_from_text(text, **kwargs)
    
    aligned_chunks = []
    current_char_index = 0

    assert len(meaningful_text) == len(timestamps), "Length of text and timestamps do not match."

    for chunk in chunks:
        chunk_text = chunk.text
        meaningful_chunk_text = ''.join([char for char in chunk_text if char.isalnum() or '\u4e00' <= char <= '\u9fff' or char.isalpha()])
        start_indices = _kmp_search(meaningful_chunk_text, meaningful_text)
        for start_index in start_indices:
            end_index = start_index + len(meaningful_chunk_text) - 1
            """
                timestamp is like [start_time, end_time], 
                using start_time as start_timestamp and end_time as end_timestamp of each chunk_timestamp
            """
            chunk_start_timestamp = timestamps[start_index][0]
            chunk_end_timestamp = timestamps[end_index][1]

            aligned_chunk = {
                "text": chunk_text,
                "start": round(chunk_start_timestamp / 1000, 2),
                "end": round(chunk_end_timestamp / 1000, 2)
            }
            aligned_chunks.append(aligned_chunk)
            current_char_index = end_index + 1

    assert current_char_index == len(timestamps), "Some characters are not aligned with any chunk."

    output_key = text_json['key']
    output = dict()
    output['key'] = output_key
    output['text'] = text
    output['chunks'] = aligned_chunks
    output_path = text_path.replace("text_3.json", "chunk_4.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return aligned_chunks


if __name__ == "__main__":
    # audio_path = "./test/audio_test.wav"
    # extract_text_from_audio(audio_path)
    text_path = "./test/text_3.json"
    aligned_chunks = align_chunks_with_timestamps(text_path, use_semantic_chunker=False)
    for chunk in aligned_chunks:
        print(chunk)