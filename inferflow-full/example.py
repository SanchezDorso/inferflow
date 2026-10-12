import argparse
import os

from transformers import AutoTokenizer

from inferflow import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, tensor_parallel_size=args.tp, enforce_eager=args.enforce_eager)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    questions = [
        "介绍一下你自己，并列举你的三个优点",
        "list all prime numbers within 100",
        "用三句话解释 毛泽东思想",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for q in questions
    ]
    outputs = llm.generate(prompts, sampling_params)
    for q, out in zip(questions, outputs):
        print("\n" + "=" * 80)
        print(f"Prompt: {q!r}")
        print(f"Completion: {out['text']!r}")
    llm.exit()


if __name__ == "__main__":
    main()
