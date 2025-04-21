import re

class CustomLatexTextSplitter:
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text):
        section_markers = [
            r'\\section{', 
            r'\\subsection{', 
            r'\\subsubsection{', 
            r'\\paragraph{', 
            r'\\subparagraph{',
            r'\\begin{document}',
            r'\\end{document}',
            r'\\maketitle'
        ]
        
        splits = []
        last_end = 0
        
        for marker in section_markers:
            pattern = re.compile(marker)
            for match in pattern.finditer(text):
                start = match.start()
                if start > last_end:
                    if start - last_end > 10:  # Avoid tiny splits
                        splits.append(text[last_end:start])
                last_end = start
        
        if last_end < len(text):
            splits.append(text[last_end:])
        
        final_splits = []
        for chunk in splits:
            if len(chunk) <= self.chunk_size:
                final_splits.append(chunk)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        final_splits.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    final_splits.append(current_chunk.strip())
        
        return final_splits

def split_latex_text(latex_text):

    splitter = CustomLatexTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_text(latex_text)
    
    print("Text splitting results:")
    for i, split in enumerate(splits):
        print(f"Split {i+1}: {split}\n")
    
    return splits

if __name__ == "__main__":
    latex_text = r"""
    \documentclass{article}

    \begin{document}

    \maketitle

    \section{Introduction}

    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.

    \subsection{History of LLMs}

    The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

    \subsection{Applications of LLMs}

    LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

    \end{document}
    """
    
    splits = split_latex_text(latex_text)
    print(f"LaTeX text split into {len(splits)} chunks") 