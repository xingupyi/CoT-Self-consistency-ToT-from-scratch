import re
import os,re
from collections import Counter
from dotenv import load_dotenv
import anthropic

GSM8K_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every egg at the farmers' market for $2. How much does she make every day at the farmers' market?",
        "reasoning": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4, using 3 + 4 = 7 eggs. So she has 16 - 7 = 9 eggs left. She sells each for $2, so she makes 9 * 2 = $18 per day.",
        "answer": "18"
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?"
        ),
        "reasoning": (
            "It takes 2 bolts of blue fiber. "
            "Half of 2 is 1, so it takes 1 bolt of white fiber. "
            "In total, 2 + 1 = 3 bolts."
        ),
        "answer": "3",
    },
    {
        "question": (
            "Josh decides to try flipping a house. He buys a house for $80,000 "
            "and puts $50,000 in repairs. This increased the value of the house "
            "by 150%. How much profit did he make?"
        ),
        "reasoning": (
            "The house cost $80,000. Repairs cost $50,000. "
            "Total investment: 80,000 + 50,000 = $130,000. "
            "The value increased by 150% of $80,000: 80,000 * 1.5 = $120,000. "
            "New value: 80,000 + 120,000 = $200,000. "
            "Profit: 200,000 - 130,000 = $70,000."
        ),
        "answer": "70000",
    },
    {
        "question": (
            "James writes a 3-page letter to 2 different friends twice a week. "
            "How many pages does he write a year?"
        ),
        "reasoning": (
            "He writes to 2 friends, so 2 letters each time. "
            "Each letter is 3 pages, so 2 * 3 = 6 pages per session. "
            "He does this twice a week: 6 * 2 = 12 pages per week. "
            "In a year (52 weeks): 12 * 52 = 624 pages."
        ),
        "answer": "624",
    },
    {
        "question": (
            "Every day, Wendi feeds each of her chickens three cups of mixed "
            "chicken feed, containing seeds, mealworms, and vegetables. She gives "
            "the chickens their feed in three separate meals. In the morning, she "
            "gives her flock of chickens 15 cups of feed. In the afternoon, she "
            "gives her chickens another 25 cups of feed. How many cups of feed "
            "does she need to give her chickens in the final meal of the day if "
            "the carry-over from prior feedings was 35 cups?"
        ),
        "reasoning": (
            "Morning feed: 15 cups. Afternoon feed: 25 cups. "
            "Total so far: 15 + 25 = 40 cups. "
            "Carry-over: 35 cups. Effective fed: 40 - 35 = 5 cups net new. "
            "Wait, let me re-read. She has a flock. Morning: 15 cups. Afternoon: 25 cups. "
            "Total given so far: 15 + 25 = 40 cups. "
            "With 35 cups carry-over, total available is 40 + 35 = 75 cups. "
            "Actually, carry-over means leftover from before. "
            "Each chicken gets 3 cups/day. Number of chickens: 15/? "
            "Morning she gives 15 cups. Each meal is 1/3 of daily feed. "
            "So 15 cups in morning = 1/3 of total daily. Total daily = 45 cups. "
            "She gave 15 + 25 = 40 cups in first two meals. "
            "Remaining: 45 - 40 = 5 cups. But carry-over is 35 cups. "
            "She needs 5 - 35 = needs to give negative? No. "
            "Total needed for last meal: the daily total minus what was already fed. "
            "15 chickens (since 15 cups / 1 cup per chicken per meal = 15 chickens). "
            "Daily total: 15 * 3 = 45 cups. Given: 15 + 25 = 40. "
            "Last meal needs: 45 - 40 = 5 cups. But the carry-over is extra, not a reduction. "
            "She needs to give 45 - 40 + 35 = 40 cups. Wait. "
            "Hmm, with 35 cups carry-over from prior feedings already counted: "
            "She needs to provide 45 - 35 = 10 total new cups today. "
            "She already gave 15 + 25 = 40. That's way more than 10. "
            "The question asks how many cups in the final meal. "
            "Let me just compute: total daily = 15 * 3 = 45. "
            "Already given: 15 + 25 = 40. Last meal: 45 - 40 = 5."
        ),
        "answer": "5",
    },
]

#llm response
load_dotenv()
client = anthropic.Anthropic()
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")

def call_llm(client, model, system, user, temperature=None):
    # The Claude API no longer accepts temperature. The argument stays only
    # so the lesson's functions can still pass it without errors.
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[
            {"role": "user", "content": user},
        ],
    )
    # content is a list of blocks; thinking blocks can come before the answer
    return "".join(b.text for b in response.content if b.type == "text")

    
#extracts number from string returned by LLM
def extract_answer(text):
    if not text:
        return None
    patterns = [
        r"[Tt]he answer is[:\s]*\$?([\d,]+(?:\.\d+)?)",
        r"#### ([\d,]+(?:\.\d+)?)",
        r"= \$?([\d,]+(?:\.\d+)?)\s*$",
    ]
    # Use regex to find the first number in the text
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")
    #fall back to last number in text if no pattern matches
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None

#-----------------------------------------------cot-----------------------------------------------
#cot prompt builder
def build_cot_prompt(question, examples, num_examples=3):
    system = (
        "You are a math problem solver. "
        "For each problem, show your step-by-step reasoning, "
        "then give the final numerical answer on the last line "
        "in the format: 'The answer is [number]'."
    )

    #provide few-shot examples from GSM8K_EXAMPLES
    example_text = ""
    for ex in examples[:num_examples]:
        example_text += f"Q: {ex['question']}\n"
        example_text += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"

    #invite model to solve the new question based on the examples
    user = f"{example_text}Q: {question}\nA:"
    return system, user

#answer question using few-shot chain-of-thought prompting
def few_shot_cot_solve(question, examples, client, model, num_examples=3):
    system,user=build_cot_prompt(question, examples, num_examples)
    text = call_llm(client, model, system, user)
    return extract_answer(text)

#-----------------------------------------------self consistency-----------------------------------------------
#self consistency voting
def self_consistency_voting(question, examples, client, model, n_examples=5):
    #each branch uses cot
    system,user=build_cot_prompt(question, examples)
    
    answers = []
    reasonings = []
    for _ in range(n_examples):
        text = call_llm(client, model, system, user)
        reasonings.append(text)
        answer = extract_answer(text)
        if answer is not None:
            answers.append(answer)
    
    if not answers:
        return None, 0.0,reasonings, Counter()
    
    #vote for the most common answer
    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[best_answer] / len(answers)
    
    return best_answer, confidence, reasonings, vote_counts

#create multiple reasoning paths for a question, each with different approach without giving the final ans
def generate_initial_thoughts(question, client, model, breadth=3):
    system = (
        "You are a math problem solver exploring different approaches to solve a problem. "
        "Generate one distinct approach to solving this problem. "
        "Show your partial reasoning, but do not give the final answer. "
    )
    thoughts = []
    
    for i in range(breadth):
        user = (
            f"Q: {question}\n\n"
            f"Approach: #{i+1} (use a diff strategy than previous approaches)."
            f"Think about: arithmetic breakdown, working backwards, algebraic representation, or any other method."
        )
        text = call_llm(client, model, system, user)
        thoughts.append(text)
    return thoughts

#prune: score each partial reasoning from 0 to 1 based on how promising it is
def evaluate_thought(thought, question, client, model):
    system = (
        "You are a math problem solver evaluating the quality of a proposed approach to solve a problem. "
        "Score the approach from 0.0 to 1.0. "
        "Provide a brief justification for your score."
        "Consider correctness of arithmetic, logical coherence, and potential to lead to a correct solution."
        "Progess towards the answer. "
        "Respond with only the score: a number between 0.0 and 1.0, with no additional text."
    )
    user = f"Q: {question}\n\nProposed Approach:\n{thought}\n\nScore (0.0-1.0):"
    text = call_llm(client, model, system, user)
    
    try:
        score = float(re.search(r"([\d.]+)", text).group(1))
        return max(0.0, min(1.0, score))  # Ensure score is within [0.0, 1.0]
    #fallback to 0.5 if score cannot be extracted
    except (AttributeError, ValueError):
       return 0.5
   
#ask the model to continue with reasoning and provide the final answer based on the partial reasoning 
def extend_thought(thought, question, client, model, breadth=2):
    system = (
       "You are a math problem solver continuing a line of reasoning. "
        "Take the partial reasoning below and extend it further toward a solution. "
        "Show your continued reasoning. If you reach the final answer, "
        "state it as: 'The answer is [number]'."
    )
    
    extensions = []
    
    for i in range(breadth):
        user = (
            f"Problem: {question}\n\n"
            f"Reasoning so far:\n{thought}\n\n"
            f"Continue this reasoning (approach #{i + 1}):"
        )
        text = call_llm(client, model, system, user)
        extensions.append(f"{thought}\n\n{text}")
    return extensions


#tree of thought
def tree_of_thought_solve(question, client, model, breadth=3, depth=3):
    #generate initial thoughts
    thoughts = generate_initial_thoughts(question, client, model, breadth)
    #score each thought and sort by score
    scored = [(t, evaluate_thought(t, question, client, model)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)

    #expand the top 2 most promising thoughts to a certain depth
    for current_depth in range(1, depth):
        next_thoughts = []
        top_k = min(2, len(scored))
        for thought, score in scored[:top_k]:
            extensions = extend_thought(thought, question, client, model, breadth)
            for ext in extensions:
                ext_score = evaluate_thought(ext, question, client, model)
                next_thoughts.append((ext, ext_score))
        #sort the next thoughts by score and keep the top 2 for the next depth level
        if next_thoughts:
            scored = sorted(next_thoughts, key=lambda x: x[1], reverse=True)

    best_thought = scored[0][0] if scored else ""
    return extract_answer(best_thought), best_thought

#solve with simplest approach first, then escalate to more complex methods if needed
def solve_with_escalation(question, examples, client, model, min_confidence=0.8):
    #single few-shot cot answer
    single_answer = few_shot_cot_solve(question, examples, client, model)

    #always run self-consistency voting to check how sure the model is
    sc_answer, confidence, reasonings, votes = self_consistency_voting(question, examples, client, model)

    #accept only if the votes mostly agree AND match the single answer
    if sc_answer is not None and confidence >= min_confidence and single_answer == sc_answer:
        return {
            "answer": sc_answer,
            "method": "self-consistency-voting",
            "confidence": confidence,
            "votes": dict(votes),
            "reasoning": reasonings[0],
        }

    #disagreement or low confidence -> escalate to tree of thought
    tot_answer, best_thought = tree_of_thought_solve(question, client, model)
    return {
        "answer": tot_answer,
        "method": "tree-of-thought",
        "confidence": confidence,  # the voting confidence that triggered escalation
        "votes": dict(votes),
        "reasoning": best_thought,
    }

TEST_QUESTIONS = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, "
            "and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?"
        ),
        "answer": "72",
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just "
            "did 50 minutes of babysitting. How much did she earn?"
        ),
        "answer": "10",
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. "
            "Betty has only half of the money she needs. Her parents decided "
            "to give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need to buy "
            "the wallet?"
        ),
        "answer": "5",
    },
    {
        "question": (
            "Julie is reading a 120-page book. Yesterday, she was able to "
            "read 12 pages and today, she read twice as many pages as yesterday. "
            "If she wants to read half of the remaining pages tomorrow, "
            "how many pages should she read?"
        ),
        "answer": "42",
    },
    {
        "question": (
            "James writes a 3-page letter to 2 different friends twice a week. "
            "How many pages does he write a year?"
        ),
        "answer": "624",
    },
]

if __name__ == "__main__":
    correct = 0
    for i, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Q: {test['question'][:80]}...")
        result = solve_with_escalation(test["question"], GSM8K_EXAMPLES, client, MODEL)

        is_correct = str(result["answer"]) == test["answer"]
        correct += is_correct

        print(f"  Method:     {result['method']}")
        print(f"  Answer:     {result['answer']} (expected: {test['answer']}) {'✓' if is_correct else '✗'}")
        # confidence/votes only exist when self-consistency voting was used
        if "confidence" in result:
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Votes:      {result['votes']}")

    print(f"\nAccuracy: {correct}/{len(TEST_QUESTIONS)}")
