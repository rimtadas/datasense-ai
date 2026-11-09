from agent.runner import run_agent


if __name__ == "__main__":
    question = (
        "Compare the number of transactions with payment_status "
        "equal to Completed versus Failed."
    )

    result = run_agent(question=question, verbose=True)

    print("\nExecution summary")
    print("-----------------")
    print("Tool calls:", result["tool_call_count"])
    print("Rounds:", result["rounds_used"])
    print("Completed:", result["completed"])
    print("Errors:", result["execution_errors"])