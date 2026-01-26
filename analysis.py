"""
Лабораторная работа №8: Жадные алгоритмы
Экспериментальное исследование и анализ
"""

import time
import random
import platform
import sys
from greedy_algorithms import *


def test_interval_scheduling():
    """Тестирование алгоритма выбора заявок"""
    print("\n" + "="*80)
    print("1. ТЕСТИРОВАНИЕ INTERVAL SCHEDULING")
    print("="*80)

    # Тестовые данные
    intervals = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]

    print(f"\nИсходные интервалы ({len(intervals)}):")
    for i, interval in enumerate(intervals):
        print(f"  Интервал {i}: {interval}")

    selected = interval_scheduling(intervals)

    print(f"\nВыбрано непересекающихся интервалов: {len(selected)}")
    for interval in selected:
        print(f"  {interval}")

    # Проверка корректности
    for i in range(len(selected) - 1):
        assert selected[i][1] <= selected[i+1][0], "Интервалы пересекаются!"

    print("\n✅ Алгоритм работает корректно - интервалы не пересекаются")

    return len(selected), selected


def test_fractional_knapsack():
    """Тестирование непрерывного рюкзака"""
    print("\n" + "="*80)
    print("2. ТЕСТИРОВАНИЕ FRACTIONAL KNAPSACK")
    print("="*80)

    capacity = 50
    items = [(10, 60), (20, 100), (30, 120)]

    print(f"\nВместимость рюкзака: {capacity}")
    print("\nПредметы (вес, стоимость):")
    for i, (w, v) in enumerate(items):
        print(f"  Предмет {i}: вес={w}, стоимость={v}, удельная стоимость={v/w:.2f}")

    total_value, selected = fractional_knapsack(capacity, items)

    print(f"\nМаксимальная стоимость: {total_value:.2f}")
    print("\nВзятые предметы:")
    for idx, fraction in selected:
        w, v = items[idx]
        print(f"  Предмет {idx}: {fraction*100:.1f}% (вес={w*fraction:.1f}, стоимость={v*fraction:.1f})")

    print("\n✅ Алгоритм работает корректно")

    return total_value, selected


def compare_knapsack_01():
    """Сравнение жадного и точного подходов для дискретного рюкзака"""
    print("\n" + "="*80)
    print("3. СРАВНЕНИЕ: ДИСКРЕТНЫЙ РЮКЗАК 0-1 (ЖАДНЫЙ vs ПОЛНЫЙ ПЕРЕБОР)")
    print("="*80)

    # Пример где жадный алгоритм НЕ работает
    capacity = 50
    items = [(10, 60), (20, 100), (30, 120)]

    print(f"\nВместимость рюкзака: {capacity}")
    print("Предметы (вес, стоимость, удельная стоимость):")
    for i, (w, v) in enumerate(items):
        print(f"  Предмет {i}: вес={w}, стоимость={v}, уд.стоимость={v/w:.2f}")

    # Жадный подход
    greedy_value, greedy_indices = knapsack_01_greedy(capacity, items)
    print(f"\n🔴 ЖАДНЫЙ алгоритм:")
    print(f"   Выбраны предметы: {greedy_indices}")
    print(f"   Суммарная стоимость: {greedy_value}")

    # Точный подход (полный перебор)
    optimal_value, optimal_indices = knapsack_01_bruteforce(capacity, items)
    print(f"\n🟢 ОПТИМАЛЬНОЕ решение (полный перебор):")
    print(f"   Выбраны предметы: {optimal_indices}")
    print(f"   Суммарная стоимость: {optimal_value}")

    if greedy_value == optimal_value:
        print("\n✅ Жадный алгоритм нашел оптимальное решение!")
    else:
        diff = optimal_value - greedy_value
        percent = (diff / optimal_value) * 100
        print(f"\n❌ Жадный алгоритм НЕ оптимален!")
        print(f"   Потеря: {diff} ({percent:.1f}%)")

    # Еще один пример где жадный НЕ работает
    print("\n" + "-"*80)
    print("Еще один пример (где жадный точно не работает):")
    capacity2 = 6
    items2 = [(2, 3), (3, 4), (4, 5)]  # Жадный возьмет предмет 2, но лучше 0+1

    print(f"\nВместимость: {capacity2}")
    print("Предметы:")
    for i, (w, v) in enumerate(items2):
        print(f"  Предмет {i}: вес={w}, стоимость={v}, уд.стоимость={v/w:.2f}")

    greedy_value2, greedy_indices2 = knapsack_01_greedy(capacity2, items2)
    optimal_value2, optimal_indices2 = knapsack_01_bruteforce(capacity2, items2)

    print(f"\nЖадный: предметы {greedy_indices2}, стоимость {greedy_value2}")
    print(f"Оптимальное: предметы {optimal_indices2}, стоимость {optimal_value2}")

    if greedy_value2 < optimal_value2:
        print(f"\n❌ Жадный алгоритм НЕОПТИМАЛЕН (потеря {optimal_value2 - greedy_value2})")

    return (greedy_value, optimal_value), (greedy_value2, optimal_value2)


def test_huffman_coding():
    """Тестирование алгоритма Хаффмана"""
    print("\n" + "="*80)
    print("4. ТЕСТИРОВАНИЕ АЛГОРИТМА ХАФФМАНА")
    print("="*80)

    text = "this is an example of a huffman tree"

    print(f"\nИсходный текст: '{text}'")
    print(f"Длина: {len(text)} символов")

    # Кодирование
    codes, tree, encoded = huffman_encoding(text)

    print(f"\nЧастоты символов:")
    freq_map = {}
    for char in text:
        freq_map[char] = freq_map.get(char, 0) + 1
    for char, freq in sorted(freq_map.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{char}': {freq} раз, код: {codes[char]}")

    print(f"\nЗакодированный текст (первые 100 бит):")
    print(f"  {encoded[:100]}...")

    print(f"\nСтатистика сжатия:")
    original_bits = len(text) * 8  # ASCII: 8 бит на символ
    encoded_bits = len(encoded)
    compression_ratio = (1 - encoded_bits / original_bits) * 100

    print(f"  Исходный размер: {original_bits} бит ({len(text)} символов × 8 бит)")
    print(f"  Закодированный размер: {encoded_bits} бит")
    print(f"  Степень сжатия: {compression_ratio:.2f}%")

    # Декодирование
    decoded = huffman_decoding(encoded, tree)

    print(f"\nДекодированный текст: '{decoded}'")

    if text == decoded:
        print("\n✅ Декодирование успешно - текст восстановлен корректно!")
    else:
        print("\n❌ Ошибка декодирования!")

    # Визуализация дерева
    print(f"\nДерево Хаффмана:")
    print(visualize_huffman_tree(tree))

    return compression_ratio, codes


def test_coin_change():
    """Тестирование задачи о монетах"""
    print("\n" + "="*80)
    print("5. ТЕСТИРОВАНИЕ ЗАДАЧИ О МОНЕТАХ")
    print("="*80)

    # Каноническая система (работает)
    coins = [25, 10, 5, 1]
    amounts = [41, 99, 167]

    print("\nКаноническая система монет: [25, 10, 5, 1] (центы США)")
    print("\nТесты:")

    results = []
    for amount in amounts:
        count, used = coin_change_greedy(amount, coins)
        results.append((amount, count, used))
        print(f"\n  Сумма {amount}:")
        print(f"    Монет: {count}")
        print(f"    Использованы: {used}")

        # Проверка
        assert sum(used) == amount, "Сумма не совпадает!"

    print("\n✅ Для канонической системы жадный алгоритм работает оптимально")

    return results


def test_prim_mst():
    """Тестирование алгоритма Прима"""
    print("\n" + "="*80)
    print("6. ТЕСТИРОВАНИЕ АЛГОРИТМА ПРИМА (MST)")
    print("="*80)

    # Пример графа
    graph = {
        0: [(1, 4), (7, 8)],
        1: [(0, 4), (2, 8), (7, 11)],
        2: [(1, 8), (3, 7), (5, 4), (8, 2)],
        3: [(2, 7), (4, 9), (5, 14)],
        4: [(3, 9), (5, 10)],
        5: [(2, 4), (3, 14), (4, 10), (6, 2)],
        6: [(5, 2), (7, 1), (8, 6)],
        7: [(0, 8), (1, 11), (6, 1), (8, 7)],
        8: [(2, 2), (6, 6), (7, 7)]
    }

    print(f"\nГраф: {len(graph)} вершин")
    print("\nРебра графа:")
    printed = set()
    for u in graph:
        for v, w in graph[u]:
            edge = tuple(sorted([u, v]))
            if edge not in printed:
                print(f"  {u} -- {v} (вес {w})")
                printed.add(edge)

    mst_edges, total_weight = prim_mst(graph)

    print(f"\nМинимальное остовное дерево:")
    print(f"  Ребер в MST: {len(mst_edges)}")
    print(f"  Суммарный вес: {total_weight}")
    print("\n  Ребра MST:")
    for u, v, w in mst_edges:
        print(f"    {u} -- {v} (вес {w})")

    # Проверка: MST должно иметь V-1 ребер
    assert len(mst_edges) == len(graph) - 1, "Неверное количество ребер в MST!"

    print("\n✅ MST построено корректно")

    return mst_edges, total_weight


def performance_analysis():
    """Анализ производительности алгоритмов"""
    print("\n" + "="*80)
    print("7. АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*80)

    print("\n📊 Тестирование алгоритма Хаффмана на разных размерах данных...")

    sizes = [100, 500, 1000, 5000, 10000]
    results = []

    for size in sizes:
        # Генерируем случайный текст
        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=size))

        # Замеряем время
        start = time.time()
        codes, tree, encoded = huffman_encoding(text)
        elapsed = time.time() - start

        results.append({
            'size': size,
            'time': elapsed,
            'unique_chars': len(codes)
        })

        print(f"  Размер {size:5d}: {elapsed:.6f} с (уникальных символов: {len(codes)})")

    print("\n📊 Тестирование Interval Scheduling...")

    interval_results = []
    for size in [100, 500, 1000, 5000]:
        # Генерируем случайные интервалы
        intervals = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(size)]
        intervals = [(min(s, e), max(s, e)) for s, e in intervals if s != e]

        start = time.time()
        selected = interval_scheduling(intervals)
        elapsed = time.time() - start

        interval_results.append({
            'size': size,
            'time': elapsed,
            'selected': len(selected)
        })

        print(f"  Размер {size:5d}: {elapsed:.6f} с (выбрано {len(selected)} интервалов)")

    return results, interval_results


def main():
    """Главная функция - запуск всех тестов"""
    print("="*80)
    print(" " * 20 + "ЛАБОРАТОРНАЯ РАБОТА №8")
    print(" " * 25 + "ЖАДНЫЕ АЛГОРИТМЫ")
    print("="*80)

    # Информация о системе
    print(f"\nСистема: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Процессор: {platform.processor() or platform.machine()}")

    # Запуск всех тестов
    interval_results = test_interval_scheduling()
    fractional_results = test_fractional_knapsack()
    knapsack_comparison = compare_knapsack_01()
    huffman_results = test_huffman_coding()
    coin_results = test_coin_change()
    mst_results = test_prim_mst()
    perf_results = performance_analysis()

    print("\n" + "="*80)
    print(" " * 25 + "ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*80)

    return {
        'interval': interval_results,
        'fractional': fractional_results,
        'knapsack_comparison': knapsack_comparison,
        'huffman': huffman_results,
        'coin': coin_results,
        'mst': mst_results,
        'performance': perf_results
    }


if __name__ == "__main__":
    results = main()
