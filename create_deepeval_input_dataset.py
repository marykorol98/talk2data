from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from core.evaluation.inference_script import METADATA
from core.evaluation.ports import WorkflowRunner
from core.models import ModelLoader
from core.workflow import WorkflowEngine

DATASET_NAME = "benchmark_v1"

# СОЗДАНИЕ ДАТАСЕТА И ПУШ

# benchmark_files = sorted(BENCHMARKS_DIR.glob("*.json"))
# goldens = []

# for path in benchmark_files:
#     benchmarks = load_json(path)

#     for benchmark in benchmarks:
#         golden_input = Golden(input=benchmark.get("user_input"))
#         # goldens are what makes up your dataset
#         goldens.append(golden_input)


# # create dataset
# dataset = EvaluationDataset(goldens=goldens)
# # save to Confident AI
# dataset.push(alias=DATASET_NAME)


class EvaluationLab:
    def __init__(self):
        loader = ModelLoader()
        workflow_engine = WorkflowEngine(loader.get_llm(), loader.get_tokenizer())
        self._workflow: WorkflowRunner = workflow_engine.create_workflow()

    @property
    def workflow(self) -> WorkflowRunner:
        if self._workflow is not None:
            return self._workflow

        raise ValueError("No workflow initialized")

    def llm_app(self, bench_input: str) -> str:
        initial_state = {
            "user_input": bench_input,
            "metadata": METADATA,
            "conversation_history": [],
            "generated_code": None,
            "response_message": None,
            "response_audio": None,
            "decision": None,
            "timing_info": {},
        }

        result_state = self.workflow.invoke(initial_state)

        decision = result_state.get("decision") or {}
        decision_action = decision.get("action") if isinstance(decision, dict) else decision

        if decision_action == "code_generation":
            return result_state.get("generated_code")

        return result_state.get("response_message")


def main():
    # Pull from Confident AI
    dataset = EvaluationDataset()
    dataset.pull(alias=DATASET_NAME)

    eval_lab = EvaluationLab()
    # Create test cases
    for golden in dataset.goldens:
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=eval_lab.llm_app(golden.input),  # Replace with your LLM app
        )
        dataset.add_test_case(test_case)

    relevancy = AnswerRelevancyMetric()  # Using this for the sake of simplicity

    # Run an evaluation
    evaluate(test_cases=dataset.test_cases, metrics=[relevancy])


if __name__ == "__main__":
    main()
