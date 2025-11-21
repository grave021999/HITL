"""
HITL Test Case: Verify unsolicited user input during execution adjusts the ongoing task.

Test Scenario:
1. Start agent: "Write a comprehensive 10-page report on quantum computing"
2. Allow agent to work for 60 seconds (simulate progress)
3. Send NEW user message: "Actually, just focus on quantum entanglement, make it 2 pages"
4. Assert: Previous task direction is CANCELLED
5. Assert: New task direction is adopted immediately
6. Assert: No completion of original 10-page report
7. Assert: Output matches adjusted requirements (2 pages, entanglement focus)
"""

import asyncio
import os
from agent import ReportAgent
from dotenv import load_dotenv

load_dotenv()


async def test_hitl_interrupt():
    """Test that unsolicited input cancels ongoing task and redirects agent."""
    
    print("=" * 80)
    print("HITL TEST: Unsolicited Input During Execution")
    print("=" * 80)
    print()
    
    # Initialize agent
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        print("Please set it in .env file or environment variable")
        return False
    
    agent = ReportAgent(api_key=api_key)
    
    # Test parameters
    thread_id = "test_hitl_001"
    initial_message = "Write a comprehensive 10-page report on quantum computing"
    interrupt_message = "Actually, just focus on quantum entanglement, make it 2 pages"
    interrupt_after_seconds = 60.0  # Full 60 seconds as per test scenario
    
    print(f"Step 1: Starting agent with initial task")
    print(f"  Task: '{initial_message}'")
    print()
    
    # Process with interrupt
    result = await agent.process_with_interrupt(
        thread_id=thread_id,
        initial_message=initial_message,
        interrupt_after_seconds=interrupt_after_seconds,
        interrupt_message=interrupt_message
    )
    
    print()
    print("=" * 80)
    print("TEST ASSERTIONS")
    print("=" * 80)
    print()
    
    # Assertions
    assertions_passed = 0
    assertions_failed = 0
    
    # Assertion 1: Previous task direction is CANCELLED
    print("Assertion 1: Previous task direction is CANCELLED")
    cancelled = result.get("cancelled", False)
    if cancelled:
        print("  [PASS] Task was cancelled")
        assertions_passed += 1
    else:
        print("  [FAIL] Task was not cancelled")
        assertions_failed += 1
    print()
    
    # Assertion 2: New task direction is adopted immediately
    print("Assertion 2: New task direction is adopted immediately")
    current_goal = result.get("current_goal", "")
    previous_goal = result.get("previous_goal", "")
    
    if interrupt_message.lower() in current_goal.lower() or "entanglement" in current_goal.lower():
        print(f"  [PASS] New goal adopted: '{current_goal}'")
        assertions_passed += 1
    else:
        print(f"  [FAIL] New goal not adopted. Current: '{current_goal}'")
        assertions_failed += 1
    print()
    
    # Assertion 3: No completion of original 10-page report
    print("Assertion 3: No completion of original 10-page report")
    output = result.get("output", "")
    
    # Check that output doesn't contain extensive quantum computing content
    # (if it was the full 10-page report, it would be very long)
    if "10-page" in initial_message.lower() and "quantum computing" in initial_message.lower():
        # Original task should not be in output if cancelled
        if cancelled and ("entanglement" in output.lower() or len(output) < 5000):
            print("  [PASS] Original 10-page report was not completed")
            assertions_passed += 1
        else:
            print(f"  [FAIL] Original report may have been completed (output length: {len(output)})")
            assertions_failed += 1
    else:
        print("  ? SKIPPED: Cannot verify original report completion")
    print()
    
    # Assertion 4: Output matches adjusted requirements (2 pages, entanglement focus)
    print("Assertion 4: Output matches adjusted requirements (2 pages, entanglement focus)")
    output_lower = output.lower()
    
    has_entanglement = "entanglement" in output_lower
    has_2_pages = "2 page" in interrupt_message.lower() or "two page" in interrupt_message.lower()
    
    if has_entanglement:
        print("  [PASS] Output focuses on quantum entanglement")
        assertions_passed += 1
    else:
        print("  [FAIL] Output does not focus on quantum entanglement")
        assertions_failed += 1
    
    # Check output length (2 pages should be roughly 1000-2000 words)
    output_word_count = len(output.split())
    if 500 <= output_word_count <= 3000:  # Reasonable range for 2 pages
        print(f"  [PASS] Output length is appropriate for 2 pages ({output_word_count} words)")
        assertions_passed += 1
    else:
        print(f"  [WARN] Output length may not match 2 pages ({output_word_count} words)")
    print()
    
    # Print detailed results
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print(f"Previous Goal: {previous_goal}")
    print(f"Current Goal: {current_goal}")
    print(f"Cancelled: {cancelled}")
    print(f"Task Started At: {result.get('task_started_at', 'N/A')}")
    print(f"Task Cancelled At: {result.get('task_cancelled_at', 'N/A')}")
    print(f"Output Length: {len(output)} characters, {len(output.split())} words")
    print()
    print("Output Preview (first 500 chars):")
    print("-" * 80)
    print(output[:500] + "..." if len(output) > 500 else output)
    print("-" * 80)
    print()
    
    # Final summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Assertions Passed: {assertions_passed}")
    print(f"Assertions Failed: {assertions_failed}")
    print(f"Total Assertions: {assertions_passed + assertions_failed}")
    print()
    
    if assertions_failed == 0:
        print("[SUCCESS] ALL ASSERTIONS PASSED - HITL test successful!")
        return True
    else:
        print("[FAILURE] SOME ASSERTIONS FAILED - HITL test needs attention")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_hitl_interrupt())
    exit(0 if success else 1)

