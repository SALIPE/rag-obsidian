import os

from docling.document_converter import DocumentConverter


def save_markdown(content, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path

def paper_to_annotation(
        paper_path: str,
        file_name:str)-> str:

    converter = DocumentConverter()
    result = converter.convert(f'{paper_path}/{file_name}')
    return result.document.export_to_markdown()



if __name__ == "__main__":

    path = "/home/salipe/Desktop/salipe-vault/Master/lecture papers/kevolve"
    filename = "Lebatteux and Diallo - 2021 - Combining a genetic algorithm and ensemble method .pdf"
    md = paper_to_annotation(paper_path=path, file_name=filename)
    save_markdown(md, f'{path}/lebatteux_kevolve.md')