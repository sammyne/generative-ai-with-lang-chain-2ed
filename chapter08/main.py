from langchain.evaluation import load_evaluator

conciseness_evaluator = load_evaluator(
    "criteria", criteria="conciseness", llm=evaluation_llm
)
conciseness_result = conciseness_evaluator.evaluate_strings(
    prediction=prediction_health, input=prompt_health
)
print("Conciseness evaluation result:", conciseness_result)
