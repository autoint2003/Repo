import copy
import re
import stanza

from nltk.tree import Tree


class SyntaxSimplifier:
    def __init__(self, lang: str = "en") -> None:
        self.nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,constituency",
            verbose=False,
        )

    def simplify_text(self, document: str) -> str:
        doc = self.nlp(document)
        result_parts = []

        in_quote = False
        for sentence in doc.sentences:
            text = sentence.text.strip()
            if not in_quote and any([punct in text for punct in ("“", '\u201c', "\u2018")]):
                in_quote = True
            if in_quote:
                result_parts.append(text)
                if any([punct in text for punct in ("”", '\u201d', "\u2019")]):
                    in_quote = False
                continue
            tree = self._stanza_to_nltk(sentence.constituency)
            result_parts.extend(self._split_coordinations(tree))

        return " ".join(result_parts)

    # ============================
    # Helper Functions
    # ============================

    def _split_coordinations(self, tree: Tree) -> list[str]:
        """Find the nearest S containing a phrase-level CC and split it."""
        pos, coord_node = self._find_coordination(tree)
        if pos is None:
            s = self._get_sentence_from_tree(tree)
            return [s[0].upper() + s[1:] if s else s]

        conjuncts = self._extract_conjuncts(coord_node)
        if len(conjuncts) <= 1:
            s = self._get_sentence_from_tree(tree)
            return [s[0].upper() + s[1:] if s else s]

        s_pos = self._find_nearest_s_ancestor(tree, pos)
        s_tree = tree if (s_pos is None or s_pos == ()) else tree[s_pos]
        rel_pos = pos if s_pos is None else pos[len(s_pos):]

        results = []
        for index, conjunct in enumerate(conjuncts):
            if index == 0:
                context_tree = s_tree
                context_rel_pos = rel_pos
            else:
                # Drop the introductory clause for non-first conjuncts so it
                # is not repeated in every generated sentence.
                context_tree, num_stripped = self._strip_introductory(s_tree)
                if num_stripped > 0 and rel_pos and rel_pos[0] >= num_stripped:
                    context_rel_pos = (rel_pos[0] - num_stripped,) + rel_pos[1:]
                else:
                    context_tree = s_tree
                    context_rel_pos = rel_pos

            new_s_tree = self._replace_at_position(context_tree, context_rel_pos, conjunct)
            # Recurse so nested coordinations are also split.
            results.extend(self._split_coordinations(new_s_tree))

        return results

    def _find_coordination(self, tree: Tree):
        """Return (position, node) of the first phrase-level node with a CC child."""
        CLAUSE_LABELS = {'ROOT', 'S', 'SBAR', 'SINV'}
        for pos in tree.treepositions():
            node = tree[pos]
            if (
                isinstance(node, Tree)
                and node.label() not in CLAUSE_LABELS
                and self._has_cc(node)
            ):
                return pos, node
        return None, None

    def _find_nearest_s_ancestor(self, tree: Tree, pos: tuple):
        """Return the position of the nearest S ancestor of pos, or None."""
        for length in range(len(pos) - 1, 0, -1):
            ancestor_pos = pos[:length]
            ancestor = tree[ancestor_pos]
            if isinstance(ancestor, Tree) and ancestor.label() == 'S':
                return ancestor_pos
        if isinstance(tree, Tree) and tree.label() == 'S':
            return ()
        return None

    def _strip_introductory(self, s_tree: Tree) -> tuple[Tree, int]:
        INTRO_LABELS = {'SBAR', 'ADVP', 'PP'}
        children = list(s_tree)
        start = 0
        if children and isinstance(children[0], Tree) and children[0].label() in INTRO_LABELS:
            start = 1
            if start < len(children) and isinstance(children[start], Tree) and children[start].label() == ',':
                start += 1
        if start == 0:
            return s_tree, 0
        return Tree(s_tree.label(), children[start:]), start

    def _has_cc(self, node: Tree) -> bool:
        return any(isinstance(child, Tree) and child.label() == 'CC' for child in node)

    def _extract_conjuncts(self, node: Tree) -> list[Tree]:
        SEPARATOR_LABELS = {'CC', ',', '.', ':', ';', "''", '``'}
        MODIFIER_LABELS = {'SBAR', 'PP', 'ADJP', 'RRC'}

        groups: list[list[Tree]] = []
        current: list[Tree] = []

        for child in node:
            if isinstance(child, Tree) and child.label() in SEPARATOR_LABELS:
                if current:
                    groups.append(current)
                    current = []
            elif isinstance(child, Tree):
                current.append(child)

        if current:
            groups.append(current)

        # Shared-head detection: "a world [in which …] and [where …]"
        if len(groups) >= 2:
            first = groups[0]
            rest = groups[1:]
            if (
                len(first) > 1
                and isinstance(first[-1], Tree)
                and first[-1].label() in MODIFIER_LABELS
                and all(
                    len(g) == 1 and isinstance(g[0], Tree) and g[0].label() in MODIFIER_LABELS
                    for g in rest
                )
            ):
                shared = first[:-1]
                groups = [[*shared, first[-1]]] + [[*shared, g[0]] for g in rest]

        conjuncts = []
        for group in groups:
            if len(group) == 1:
                conjuncts.append(group[0])
            else:
                conjuncts.append(Tree(node.label(), group))

        return conjuncts

    def _replace_at_position(self, tree: Tree, position: tuple, new_subtree) -> Tree:
        """Return a deep copy of tree with the node at position replaced."""
        new_tree = copy.deepcopy(tree)
        if len(position) == 0:
            return new_subtree
        node = new_tree
        for idx in position[:-1]:
            node = node[idx]
        node[position[-1]] = new_subtree
        return new_tree

    def _splice_at_position(self, tree: Tree, position: tuple, new_subtrees: list) -> Tree:
        """Return a deep copy of tree with the node at position replaced by
        multiple subtrees spliced into the parent's child list."""
        new_tree = copy.deepcopy(tree)
        node = new_tree
        for idx in position[:-1]:
            node = node[idx]
        idx = position[-1]
        node[idx : idx + 1] = new_subtrees
        return new_tree

    def _stanza_to_nltk(self, stanza_tree) -> Tree:
        """Converts a Stanza parse tree to an NLTK Tree."""
        return Tree.fromstring(str(stanza_tree))

    def _get_sentence_from_tree(self, tree) -> str:
        """Flattens the tree back into a string and cleans up basic punctuation."""
        words = tree.leaves()
        sentence = " ".join(words)
        sentence = re.sub(r"\s+([.,?!:;])", r"\1", sentence)
        sentence = re.sub(r"\s+(['’]s|['’]re|['’]ve|['’]ll|['’]d|['’]m|n['’]t)\b", r"\1", sentence)
        sentence = re.sub(r"\s+-\s+", "-", sentence)
        sentence = re.sub(r"-LRB-\s*", "(", sentence)
        sentence = re.sub(r"\s*-RRB-", ")", sentence)
        if sentence and sentence[-1] not in ".,!?":
            sentence += "."

        return sentence
    
if __name__ == "__main__":
    import csv
    
    # Load row 3 (0-indexed) from CSV
    csv_path = "./data/cna_articles.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        row_3 = rows[3]
    
    simplifier = SyntaxSimplifier()
    text = row_3["body_content"]
    print(f"Title: {row_3['title']}\n")
    print(f"Input text length: {len(text)} characters\n")
    print("=" * 80)
    print("ORIGINAL TEXT:")
    print("=" * 80)
    print(text)
    print("\n" + "=" * 80)
    print("SIMPLIFIED OUTPUT:")
    print("=" * 80)
    simplified = simplifier.simplify_text(text)
    print(simplified)