"""
Test script for Exercise 1: Hotel Details with RAG

This script tests the RAG agent without running the WebSocket server.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from agents.hotel_rag_agent import answer_hotel_question_rag


def test_exercise_1():
    print("🧪 Testing Exercise 1: RAG Hotel Agent")
    print("=" * 60)

    test_queries = [
        "List all hotels in France",
        "What is the full address of Grand Victoria?",
        "What meal plans are available in Paris hotels?",
        "Compare triple room prices in Nice for off season",
        "What is the discount for extra bed in Obsidian Tower?",
    ]

    success = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 60)

        try:
            response = answer_hotel_question_rag(query)
            print("✅ Response:")
            print(response[:500])  # Show first 500 chars
            success += 1
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Results: {success}/{len(test_queries)} tests passed")

    if success == len(test_queries):
        print("✅ Exercise 1 RAG agent working correctly!")
    else:
        print("⚠️ Some tests failed. Check logs.")


if __name__ == "__main__":
    test_exercise_1()
