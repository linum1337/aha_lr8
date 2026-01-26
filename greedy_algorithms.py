"""
Лабораторная работа №8: Жадные алгоритмы
Реализация классических жадных алгоритмов
"""

from typing import List, Tuple, Dict
import heapq


# ============================================================================
# 1. ЗАДАЧА О ВЫБОРЕ ЗАЯВОК (INTERVAL SCHEDULING)
# ============================================================================

def interval_scheduling(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Выбирает максимальное количество непересекающихся интервалов.

    Жадная стратегия: Сортировка по времени окончания, выбор интервалов,
    которые заканчиваются раньше всех и не пересекаются с уже выбранными.

    Временная сложность: O(n log n) - доминирует сортировка
    Пространственная сложность: O(n) - для хранения результата

    Args:
        intervals: Список кортежей (start, end) - интервалы времени

    Returns:
        Список выбранных непересекающихся интервалов

    Пример:
        >>> interval_scheduling([(1, 3), (2, 5), (4, 6), (6, 7)])
        [(1, 3), (4, 6), (6, 7)]
    """
    if not intervals:
        return []

    # Сортируем по времени окончания (жадный выбор)
    # O(n log n)
    sorted_intervals = sorted(intervals, key=lambda x: x[1])

    selected = [sorted_intervals[0]]
    last_end = sorted_intervals[0][1]

    # O(n)
    for start, end in sorted_intervals[1:]:
        # Если интервал не пересекается с последним выбранным
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected


# ============================================================================
# 2. НЕПРЕРЫВНЫЙ РЮКЗАК (FRACTIONAL KNAPSACK)
# ============================================================================

def fractional_knapsack(capacity: float, items: List[Tuple[float, float]]) -> Tuple[float, List[Tuple[int, float]]]:
    """
    Решает задачу о непрерывном рюкзаке (можно брать дробные части предметов).

    Жадная стратегия: Сортировка по удельной стоимости (цена/вес),
    берем предметы с максимальной удельной стоимостью первыми.

    Временная сложность: O(n log n) - сортировка
    Пространственная сложность: O(n)

    Args:
        capacity: Вместимость рюкзака
        items: Список кортежей (weight, value) - вес и стоимость предметов

    Returns:
        (total_value, selected_items):
        - total_value: максимальная суммарная стоимость
        - selected_items: список (индекс, доля) взятых предметов

    Пример:
        >>> fractional_knapsack(50, [(10, 60), (20, 100), (30, 120)])
        (240.0, [(0, 1.0), (1, 1.0), (2, 0.667)])
    """
    if capacity <= 0 or not items:
        return 0.0, []

    # Вычисляем удельную стоимость для каждого предмета
    # и создаем список (индекс, вес, стоимость, удельная_стоимость)
    indexed_items = []
    for i, (weight, value) in enumerate(items):
        if weight > 0:
            unit_value = value / weight
            indexed_items.append((i, weight, value, unit_value))

    # Сортируем по убыванию удельной стоимости (жадный выбор)
    # O(n log n)
    indexed_items.sort(key=lambda x: x[3], reverse=True)

    total_value = 0.0
    selected_items = []
    remaining_capacity = capacity

    # O(n)
    for idx, weight, value, unit_value in indexed_items:
        if remaining_capacity == 0:
            break

        if weight <= remaining_capacity:
            # Берем весь предмет
            selected_items.append((idx, 1.0))
            total_value += value
            remaining_capacity -= weight
        else:
            # Берем часть предмета
            fraction = remaining_capacity / weight
            selected_items.append((idx, fraction))
            total_value += value * fraction
            remaining_capacity = 0

    return total_value, selected_items


# ============================================================================
# 3. ДИСКРЕТНЫЙ РЮКЗАК 0-1 (для сравнения с жадным подходом)
# ============================================================================

def knapsack_01_bruteforce(capacity: int, items: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """
    Решает дискретную задачу о рюкзаке 0-1 полным перебором (для маленьких данных).

    Временная сложность: O(2^n) - экспоненциальная
    Пространственная сложность: O(n)

    Args:
        capacity: Вместимость рюкзака
        items: Список кортежей (weight, value)

    Returns:
        (max_value, selected_indices): максимальная стоимость и индексы выбранных предметов
    """
    n = len(items)
    max_value = 0
    best_selection = []

    # Перебираем все 2^n подмножеств
    for mask in range(1 << n):
        current_weight = 0
        current_value = 0
        current_selection = []

        for i in range(n):
            if mask & (1 << i):
                current_weight += items[i][0]
                current_value += items[i][1]
                current_selection.append(i)

        if current_weight <= capacity and current_value > max_value:
            max_value = current_value
            best_selection = current_selection

    return max_value, best_selection


def knapsack_01_greedy(capacity: int, items: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
    """
    НЕОПТИМАЛЬНЫЙ жадный алгоритм для дискретного рюкзака 0-1.
    Используется для демонстрации, что жадный подход не всегда работает.

    Временная сложность: O(n log n)

    Args:
        capacity: Вместимость рюкзака
        items: Список кортежей (weight, value)

    Returns:
        (total_value, selected_indices): стоимость и индексы (может быть неоптимально!)
    """
    indexed_items = []
    for i, (weight, value) in enumerate(items):
        if weight > 0:
            unit_value = value / weight
            indexed_items.append((i, weight, value, unit_value))

    indexed_items.sort(key=lambda x: x[3], reverse=True)

    total_value = 0
    selected_indices = []
    remaining_capacity = capacity

    for idx, weight, value, _ in indexed_items:
        if weight <= remaining_capacity:
            selected_indices.append(idx)
            total_value += value
            remaining_capacity -= weight

    return total_value, selected_indices


# ============================================================================
# 4. АЛГОРИТМ ХАФФМАНА (HUFFMAN CODING)
# ============================================================================

class HuffmanNode:
    """Узел дерева Хаффмана"""
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char      # Символ (None для внутренних узлов)
        self.freq = freq      # Частота
        self.left = left      # Левый потомок
        self.right = right    # Правый потомок

    def __lt__(self, other):
        # Для сравнения в куче
        return self.freq < other.freq

    def is_leaf(self):
        return self.left is None and self.right is None


def huffman_encoding(text: str) -> Tuple[Dict[str, str], HuffmanNode, str]:
    """
    Строит оптимальный префиксный код Хаффмана для заданного текста.

    Жадная стратегия: На каждом шаге объединяем два узла с наименьшими частотами.

    Временная сложность: O(n log n), где n - количество уникальных символов
    Пространственная сложность: O(n)

    Args:
        text: Исходный текст для кодирования

    Returns:
        (codes, tree, encoded_text):
        - codes: словарь {символ: код}
        - tree: корень дерева Хаффмана
        - encoded_text: закодированный текст

    Пример:
        >>> huffman_encoding("aabbbcccc")
        ({'c': '0', 'b': '10', 'a': '11'}, <HuffmanNode>, '1111101010000')
    """
    if not text:
        return {}, None, ""

    # Подсчитываем частоты символов - O(n)
    freq_map = {}
    for char in text:
        freq_map[char] = freq_map.get(char, 0) + 1

    # Особый случай: один уникальный символ
    if len(freq_map) == 1:
        char = list(freq_map.keys())[0]
        return {char: '0'}, HuffmanNode(char, freq_map[char]), '0' * len(text)

    # Создаем приоритетную очередь (min-heap) из листовых узлов
    # O(n log n)
    heap = []
    for char, freq in freq_map.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, node)

    # Строим дерево Хаффмана (жадный выбор)
    # O(n log n) - n итераций, каждая операция с кучей O(log n)
    while len(heap) > 1:
        # Извлекаем два узла с минимальными частотами
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Создаем новый внутренний узел
        merged = HuffmanNode(
            char=None,
            freq=left.freq + right.freq,
            left=left,
            right=right
        )

        heapq.heappush(heap, merged)

    # Корень дерева
    root = heap[0]

    # Генерируем коды для каждого символа - O(n)
    codes = {}

    def generate_codes(node, code=""):
        if node.is_leaf():
            codes[node.char] = code
        else:
            if node.left:
                generate_codes(node.left, code + "0")
            if node.right:
                generate_codes(node.right, code + "1")

    generate_codes(root)

    # Кодируем текст - O(m), где m - длина текста
    encoded_text = ''.join(codes[char] for char in text)

    return codes, root, encoded_text


def huffman_decoding(encoded_text: str, root: HuffmanNode) -> str:
    """
    Декодирует текст, закодированный алгоритмом Хаффмана.

    Временная сложность: O(m), где m - длина закодированного текста

    Args:
        encoded_text: Закодированная строка битов
        root: Корень дерева Хаффмана

    Returns:
        Декодированный текст
    """
    if not encoded_text or not root:
        return ""

    # Особый случай: один символ
    if root.is_leaf():
        return root.char * len(encoded_text)

    decoded = []
    current = root

    for bit in encoded_text:
        # Идем влево или вправо по дереву
        if bit == '0':
            current = current.left
        else:
            current = current.right

        # Достигли листа - нашли символ
        if current.is_leaf():
            decoded.append(current.char)
            current = root

    return ''.join(decoded)


# ============================================================================
# 5. ЗАДАЧА О МОНЕТАХ (COIN CHANGE - GREEDY)
# ============================================================================

def coin_change_greedy(amount: int, coins: List[int]) -> Tuple[int, List[int]]:
    """
    Решает задачу о минимальном количестве монет для выдачи суммы (жадный подход).

    ВАЖНО: Жадный алгоритм работает корректно только для канонических систем монет
    (например, [1, 5, 10, 25] - центы США). Для других систем может давать неоптимальный результат.

    Жадная стратегия: Берем максимальную монету, которая не превышает оставшуюся сумму.

    Временная сложность: O(n * amount/min_coin), где n - количество номиналов
    Пространственная сложность: O(n)

    Args:
        amount: Сумма для выдачи
        coins: Список номиналов монет (должен быть отсортирован по убыванию)

    Returns:
        (count, used_coins): количество монет и список использованных монет

    Пример:
        >>> coin_change_greedy(41, [25, 10, 5, 1])
        (5, [25, 10, 5, 1])
    """
    if amount == 0:
        return 0, []

    # Сортируем монеты по убыванию
    coins_sorted = sorted(coins, reverse=True)

    used_coins = []
    remaining = amount

    for coin in coins_sorted:
        while remaining >= coin:
            used_coins.append(coin)
            remaining -= coin

    return len(used_coins), used_coins


# ============================================================================
# 6. АЛГОРИТМ ПРИМА (MINIMUM SPANNING TREE)
# ============================================================================

def prim_mst(graph: Dict[int, List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Находит минимальное остовное дерево алгоритмом Прима.

    Жадная стратегия: На каждом шаге добавляем ребро минимального веса,
    соединяющее дерево с новой вершиной.

    Временная сложность: O((V + E) log V) с приоритетной очередью
    Пространственная сложность: O(V + E)

    Args:
        graph: Граф в виде списка смежности {вершина: [(сосед, вес), ...]}

    Returns:
        (mst_edges, total_weight):
        - mst_edges: список ребер (u, v, weight) в MST
        - total_weight: суммарный вес MST

    Пример:
        >>> graph = {0: [(1, 4), (2, 3)], 1: [(0, 4), (2, 1)], 2: [(0, 3), (1, 1)]}
        >>> prim_mst(graph)
        ([(0, 2, 3), (2, 1, 1)], 4)
    """
    if not graph:
        return [], 0

    # Начинаем с произвольной вершины
    start = next(iter(graph))
    visited = {start}
    mst_edges = []
    total_weight = 0

    # Приоритетная очередь: (вес, вершина_из, вершина_в)
    heap = []
    for neighbor, weight in graph[start]:
        heapq.heappush(heap, (weight, start, neighbor))

    while heap and len(visited) < len(graph):
        weight, u, v = heapq.heappop(heap)

        if v in visited:
            continue

        # Добавляем ребро в MST
        visited.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight

        # Добавляем новые ребра в очередь
        for neighbor, edge_weight in graph[v]:
            if neighbor not in visited:
                heapq.heappush(heap, (edge_weight, v, neighbor))

    return mst_edges, total_weight


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def visualize_huffman_tree(root: HuffmanNode, prefix="", is_tail=True) -> str:
    """
    Визуализирует дерево Хаффмана в текстовом виде.

    Args:
        root: Корень дерева
        prefix: Префикс для отображения
        is_tail: Является ли узел последним в списке детей

    Returns:
        Строковое представление дерева
    """
    if not root:
        return ""

    result = prefix
    result += "└── " if is_tail else "├── "

    if root.is_leaf():
        result += f"'{root.char}' (freq={root.freq})\n"
    else:
        result += f"[{root.freq}]\n"

        if root.left or root.right:
            if root.left:
                extension = "    " if is_tail else "│   "
                result += visualize_huffman_tree(root.left, prefix + extension, root.right is None)
            if root.right:
                extension = "    " if is_tail else "│   "
                result += visualize_huffman_tree(root.right, prefix + extension, True)

    return result
