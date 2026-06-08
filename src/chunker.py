from dataclasses import dataclass

import tree_sitter
import tree_sitter_python

from src.github_client import RepoFile

_CHUNK_NODE_TYPES: frozenset[str] = frozenset({
    "function_definition",
    "class_definition",
    "decorated_definition",
})


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    text: str


def _build_parser() -> tree_sitter.Parser:
    language = tree_sitter.Language(tree_sitter_python.language())
    return tree_sitter.Parser(language)


_PARSER: tree_sitter.Parser = _build_parser()


def chunk_files(files: list[RepoFile]) -> list[Chunk]:
    result: list[Chunk] = []
    for repo_file in files:
        result.extend(_chunk_file(repo_file))
    return result


def _chunk_file(repo_file: RepoFile) -> list[Chunk]:
    source_lines: list[str] = repo_file.content.splitlines()
    tree = _PARSER.parse(repo_file.content.encode())
    root = tree.root_node

    chunks: list[Chunk] = []
    module_level_nodes: list[tree_sitter.Node] = []

    for child in root.children:
        if child.type in _CHUNK_NODE_TYPES:
            start_line: int = child.start_point.row + 1
            end_line: int = child.end_point.row + 1
            text: str = "\n".join(source_lines[start_line - 1 : end_line])
            chunks.append(
                Chunk(
                    file_path=repo_file.path,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                )
            )
        else:
            module_level_nodes.append(child)

    if module_level_nodes:
        ml_start: int = module_level_nodes[0].start_point.row + 1
        ml_end: int = module_level_nodes[-1].end_point.row + 1
        ml_text: str = "\n".join(
            "\n".join(source_lines[node.start_point.row : node.end_point.row + 1])
            for node in module_level_nodes
        )
        chunks.insert(
            0,
            Chunk(
                file_path=repo_file.path,
                start_line=ml_start,
                end_line=ml_end,
                text=ml_text,
            ),
        )

    return chunks
