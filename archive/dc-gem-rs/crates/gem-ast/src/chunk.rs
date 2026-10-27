//! Symbol-boundary chunking via tree-sitter (Python + Rust).

use tree_sitter::{Node, Parser, Tree};

/// One symbol-anchored source chunk.
#[derive(Clone, Debug)]
pub struct SymbolChunk {
    pub path: String,
    pub symbol: String,
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Split `source_code` on symbol boundaries; each block gets
/// `# [FILE: <path> | SYMBOL: <name>]` header.
///
/// Supports Python (`def`/`class`) and Rust (`fn`/`impl`/`struct`/`trait`).
/// `max_lines` caps oversized class bodies by preferring method slices.
pub fn chunk_file_by_symbols(file_path: &str, source_code: &str, max_lines: usize) -> Vec<SymbolChunk> {
    let ext = file_path.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "py" | "pyi" => chunk_python(file_path, source_code, max_lines),
        "rs" => chunk_rust(file_path, source_code, max_lines),
        _ => {
            // heuristic fallback
            if source_code.contains("def ") || source_code.contains("class ") {
                chunk_python(file_path, source_code, max_lines)
            } else if source_code.contains("fn ") || source_code.contains("impl ") {
                chunk_rust(file_path, source_code, max_lines)
            } else {
                vec![module_chunk(file_path, source_code)]
            }
        }
    }
}

fn module_chunk(path: &str, source: &str) -> SymbolChunk {
    let lines = source.lines().count().max(1);
    SymbolChunk {
        path: path.to_string(),
        symbol: "<module>".into(),
        text: format!("# [FILE: {path} | SYMBOL: <module>]\n{source}"),
        start_line: 1,
        end_line: lines,
    }
}

fn header(path: &str, sym: &str, body: &str) -> String {
    let mut b = body.to_string();
    if !b.ends_with('\n') {
        b.push('\n');
    }
    format!("# [FILE: {path} | SYMBOL: {sym}]\n{b}")
}

fn line_of(byte: usize, source: &str) -> usize {
    source[..byte.min(source.len())].bytes().filter(|&c| c == b'\n').count() + 1
}

fn slice_lines(source: &str, start_byte: usize, end_byte: usize) -> String {
    let s = start_byte.min(source.len());
    let e = end_byte.min(source.len());
    source[s..e].to_string()
}

fn child_named<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut c = node.walk();
    for ch in node.children(&mut c) {
        if ch.kind() == kind {
            return Some(ch);
        }
    }
    None
}

fn ident_text(node: Node<'_>, source: &str) -> String {
    let s = node.start_byte().min(source.len());
    let e = node.end_byte().min(source.len());
    source[s..e].to_string()
}

fn parse_lang(source: &str, language: tree_sitter::Language) -> Option<Tree> {
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    parser.parse(source, None)
}

fn chunk_python(path: &str, source: &str, max_lines: usize) -> Vec<SymbolChunk> {
    let Some(tree) = parse_lang(source, tree_sitter_python::LANGUAGE.into()) else {
        return vec![module_chunk(path, source)];
    };
    let root = tree.root_node();
    let mut spans: Vec<(String, usize, usize)> = Vec::new();

    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        match child.kind() {
            "function_definition" | "async_function_definition" => {
                if let Some(name) = child_named(child, "identifier") {
                    spans.push((
                        ident_text(name, source),
                        child.start_byte(),
                        child.end_byte(),
                    ));
                }
            }
            "class_definition" => {
                let cname = child_named(child, "identifier")
                    .map(|n| ident_text(n, source))
                    .unwrap_or_else(|| "Class".into());
                spans.push((cname.clone(), child.start_byte(), child.end_byte()));
                // methods
                let mut c2 = child.walk();
                for m in child.children(&mut c2) {
                    if m.kind() == "block" {
                        let mut c3 = m.walk();
                        for meth in m.children(&mut c3) {
                            if matches!(
                                meth.kind(),
                                "function_definition" | "async_function_definition"
                            ) {
                                if let Some(name) = child_named(meth, "identifier") {
                                    spans.push((
                                        format!("{cname}.{}", ident_text(name, source)),
                                        meth.start_byte(),
                                        meth.end_byte(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if spans.is_empty() {
        return vec![module_chunk(path, source)];
    }
    finalize_spans(path, source, spans, max_lines)
}

fn chunk_rust(path: &str, source: &str, max_lines: usize) -> Vec<SymbolChunk> {
    let Some(tree) = parse_lang(source, tree_sitter_rust::LANGUAGE.into()) else {
        return vec![module_chunk(path, source)];
    };
    let root = tree.root_node();
    let mut spans: Vec<(String, usize, usize)> = Vec::new();

    fn walk_rust(node: Node<'_>, source: &str, spans: &mut Vec<(String, usize, usize)>, prefix: &str) {
        match node.kind() {
            "function_item" => {
                if let Some(name) = child_named(node, "identifier") {
                    let n = ident_text(name, source);
                    let sym = if prefix.is_empty() {
                        n
                    } else {
                        format!("{prefix}::{n}")
                    };
                    spans.push((sym, node.start_byte(), node.end_byte()));
                }
            }
            "struct_item" | "trait_item" | "enum_item" => {
                if let Some(name) = child_named(node, "type_identifier") {
                    let n = ident_text(name, source);
                    spans.push((n, node.start_byte(), node.end_byte()));
                }
            }
            "impl_item" => {
                // type name
                let mut type_name = "impl".to_string();
                let mut c = node.walk();
                for ch in node.children(&mut c) {
                    if ch.kind() == "type_identifier" {
                        type_name = ident_text(ch, source);
                        break;
                    }
                }
                spans.push((
                    format!("impl {type_name}"),
                    node.start_byte(),
                    node.end_byte(),
                ));
                let mut c2 = node.walk();
                for ch in node.children(&mut c2) {
                    if ch.kind() == "declaration_list" {
                        let mut c3 = ch.walk();
                        for m in ch.children(&mut c3) {
                            walk_rust(m, source, spans, &type_name);
                        }
                    }
                }
            }
            _ => {
                let mut c = node.walk();
                for ch in node.children(&mut c) {
                    walk_rust(ch, source, spans, prefix);
                }
            }
        }
    }

    walk_rust(root, source, &mut spans, "");
    if spans.is_empty() {
        return vec![module_chunk(path, source)];
    }
    finalize_spans(path, source, spans, max_lines)
}

fn finalize_spans(
    path: &str,
    source: &str,
    spans: Vec<(String, usize, usize)>,
    max_lines: usize,
) -> Vec<SymbolChunk> {
    let mut seen = std::collections::HashSet::new();
    let mut chunks = Vec::new();
    for (sym, start, end) in spans {
        if !seen.insert((start, end)) {
            continue;
        }
        let body = slice_lines(source, start, end);
        let start_line = line_of(start, source);
        let end_line = line_of(end.saturating_sub(1), source);
        let nlines = end_line.saturating_sub(start_line) + 1;
        // Prefer method-sized chunks when class exceeds max_lines
        if nlines > max_lines && !sym.contains('.') && !sym.contains("::") {
            continue;
        }
        chunks.push(SymbolChunk {
            path: path.to_string(),
            symbol: sym.clone(),
            text: header(path, &sym, &body),
            start_line,
            end_line,
        });
    }
    if chunks.is_empty() {
        vec![module_chunk(path, source)]
    } else {
        chunks
    }
}
