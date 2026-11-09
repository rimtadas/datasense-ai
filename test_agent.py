from agent.runner import run_agent


if __name__ == "__main__":

    question = (
        "Among cancelled transactions, which payment methods "
        "are most common? Show the count and percentage "
        "for each payment method."
    )

    result = run_agent(
        question=question,
        verbose=True,
    )

    print("\nExecution summary")
    print("-----------------")
    print("Tool calls:", result["tool_call_count"])
    print("Rounds:", result["rounds_used"])
    print("Completed:", result["completed"])
    print("Errors:", result["execution_errors"])