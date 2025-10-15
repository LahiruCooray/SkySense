"""
Quick test script to verify copilot components
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.copilot.knowledge_builder import KnowledgeBaseBuilder
        print("  ✓ knowledge_builder")
    except ImportError as e:
        print(f"  ✗ knowledge_builder: {e}")
        return False
    
    try:
        from src.copilot.structured_query import InsightQueryEngine
        print("  ✓ structured_query")
    except ImportError as e:
        print(f"  ✗ structured_query: {e}")
        return False
    
    # RAG imports (might fail if deps not installed)
    try:
        from src.copilot.rag_engine import SkySenseRAG
        print("  ✓ rag_engine")
    except ImportError as e:
        print(f"  ⚠ rag_engine (install dependencies): {e}")
    
    try:
        from src.copilot.cli import SkySenseCopilot
        print("  ✓ cli")
    except ImportError as e:
        print(f"  ⚠ cli (install dependencies): {e}")
    
    return True


def test_knowledge_builder():
    """Test knowledge base building"""
    print("\nTesting knowledge builder...")
    
    from src.copilot.knowledge_builder import KnowledgeBaseBuilder
    
    builder = KnowledgeBaseBuilder()
    
    # Build detector knowledge
    detector_specs = builder.build_detector_knowledge()
    assert len(detector_specs) > 0, "No detector specs"
    print(f"  ✓ Built {len(detector_specs)} detector specs")
    
    # Build terminology
    terms = builder.build_terminology_glossary()
    assert len(terms) > 0, "No terminology"
    print(f"  ✓ Built {len(terms)} terms")
    
    # Check attributions
    assert "detector_specs" in builder.knowledge["attributions"]
    assert "terminology" in builder.knowledge["attributions"]
    print("  ✓ Attributions present")
    
    return True


def test_structured_query():
    """Test structured query engine"""
    print("\nTesting structured query engine...")
    
    from src.copilot.structured_query import InsightQueryEngine
    
    engine = InsightQueryEngine()
    
    # Test methods exist
    assert hasattr(engine, 'get_insights_by_type')
    assert hasattr(engine, 'get_critical_events')
    assert hasattr(engine, 'count_by_type')
    print("  ✓ All methods present")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("SkySense Copilot Component Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Knowledge Builder", test_knowledge_builder),
        ("Structured Query", test_structured_query),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("  1. Install full dependencies: pip install -r requirements.txt")
        print("  2. Set up knowledge base: python main.py setup-copilot")
        print("  3. Start copilot: python main.py copilot")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
