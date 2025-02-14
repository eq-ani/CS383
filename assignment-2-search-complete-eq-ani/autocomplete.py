from collections import deque
import heapq
import random
import string

class Node:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = {}  # Store frequency of child nodes

class Autocomplete:
    def __init__(self, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs # Default suggest method
        if document:
            self.build_tree(document)
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node.frequency[char] = node.frequency.get(char, 0) + 1
                node = node.children[char]
            node.is_word = True
    
    def suggest_bfs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []
        
        queue = deque([(node, prefix)])
        suggestions = []
        
        while queue:
            curr_node, curr_prefix = queue.popleft()
            if curr_node.is_word:
                suggestions.append(curr_prefix)
            for char, next_node in curr_node.children.items():
                queue.append((next_node, curr_prefix + char))
        
        return suggestions

    def suggest_dfs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []
        
        stack = [(node, prefix)]
        suggestions = []
        
        while stack:
            curr_node, curr_prefix = stack.pop()
            if curr_node.is_word:
                suggestions.append(curr_prefix)
            for char, next_node in sorted(curr_node.children.items(), reverse=True):
                stack.append((next_node, curr_prefix + char))
        
        return suggestions

    def suggest_ucs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []

        priority_queue = [(0, prefix, node)]  # (cost, word, node)
        suggestions = []

        while priority_queue:
            cost, curr_prefix, curr_node = heapq.heappop(priority_queue)

            if curr_node.is_word:
                suggestions.append(curr_prefix)
            
            for char, next_node in curr_node.children.items():
                new_cost = cost + (1 / curr_node.frequency.get(char, 1))  # FIXED
                heapq.heappush(priority_queue, (new_cost, curr_prefix + char, next_node))

        return suggestions


    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
