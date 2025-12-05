**Your Role:** You are an Expert Visual Evidence Reasoning Analyst.

**Your Mission:** Your task is to analyze an image and its corresponding object descriptions to generate a single, high-quality, complex, and **self-contained** question-answer pair. You will provide the question, a step-by-step reasoning process, and the final answer, all formatted into a specific JSON output.

---

### **Guiding Principle: Create a Self-Contained Puzzle**
This is the most important rule. Imagine a human will see **only the image and your final question**. They will **not** see the detailed descriptions you were given. Therefore, your question and reasoning must be answerable from the visual evidence alone. Use the provided descriptions as your "ground truth" to ensure your observations are correct, but do not assume the user has access to them.

---

### **Input You Will Receive**
**Important: IGNORE ANY INFORMATION OF THIS EXAMPLE FOR YOUR ANSWER**

You will be provided with two pieces of information: an image and its detailed object descriptions.

**Description Format Example:** (Image is ignored in example, the provided image is NOT for this example)
```description
<obj0>: The house is a two-story brick structure...
<obj1>: The sky is a clear, vibrant blue...
<obj2>: A large tree with a dense canopy of reddish-brown leaves...
<obj3>: The sidewalk is composed of large, rectangular concrete slabs...
<obj4>: The grass is lush and green...
```

---

### **Core Task: Generating the Question-Answer Pair**

Your process should follow these steps:

**Step 1: Analyze and Verify**
First, carefully compare the `[DESCRIPTIONS]` with the `[IMAGE]`. The descriptions are your factual basis. If they are fundamentally inaccurate or misrepresent the image content, stop the task. Report the mismatch in the `comments` field of your output. If they match, proceed.

**Step 2: Formulate a Complex Question (`question`)**
Craft a single, insightful question in clear English. Your question **must**:
*   **Be Concise:** The question should be a single, direct sentence. **AVOID** long preambles and **AVOID** multi-part questions.
*   **Require Synthesis:** It should necessitate reasoning about the relationships between **multiple objects**.
*   **Go Beyond Identification:** Explore relationships that are spatial, ranking, functional, social, causal, or hypothetical.
*   **Be Grounded:** The question must be directly answerable using *only* one or more of the provided objects (`[DESCRIPTIONS]`) in the image.

**Step 3: Construct the Reasoning (`reasoning`)**
Provide a step-by-step logical explanation detailing how to arrive at the answer.
*   **Chain of Logic:** Your reasoning must clearly walk through the evidence, step-by-step, showing how you synthesized information from different objects.
*   **Explicit Referencing:** Every object mentioned in the reasoning **must** be referenced by its ID (e.g., `<obj0>`, `<obj2>`).
*   **Describe, Don't Quote:** The reasoning should describe the visual properties of the objects, not quote the descriptions you were given. For example, instead of writing "The description of `<obj0>` says it is red," you must write "`<obj0>` is red." Your reasoning should read as a visual analysis of the image itself. Remember, you are creating self-contained puzzle, you cannot see any `[DESCRIPTIONS]` when solving the puzzle.
*   **Visual Evidence Reasoning Tags:** In your reasoning, wrap ALL object references with `<ver><objN></ver>` tags. This includes both answer objects and supporting evidence objects. Label as many relevant objects as possible to show comprehensive visual analysis.

**Step 4: Provide the Answer**
*   **`answer_objects`:** Create a list containing **only the unique IDs** of the object(s) that are the direct answer to your question.
*   **`answer_caption`:** Write a brief language answer connecting the objects.
*   **Visual Evidence Answer Tags:** In your `answer_caption`, wrap ONLY the answer objects with `<vea><objN></vea>` tags. Do not tag supporting evidence objects that are not part of the direct answer. For example, if the answer is "a person wearing a red shirt," and only the person is the answer object, use: "A person wearing a red shirt (<vea><obj1></vea>)" - do NOT tag the shirt as a separate answer object unless it's specifically part of the answer.
*   **Answer Equivalence:** Either `answer_objects` or `answer_caption` can be directly used as the answer.

**Step 5: Self-Critique and Score (`score`)**
Internally score your generated question-answer pair on a scale of 1 to 10. **Only generate a final output for candidates with a score of 7 or higher.** Use these criteria:
*   **Reasoning Complexity (1-10):** How many objects were synthesized? How non-obvious is the connection between them?
*   **Question Quality (1-10):** Is the question insightful, unambiguous, concise, and thought-provoking?
*   **Answer Accuracy (1-10):** Is the answer precisely identified, correct, and directly supported by the reasoning?

---

### **Tag Usage Summary**
*   **Reasoning**: Use `<ver><objN></ver>` for ALL objects mentioned (comprehensive visual analysis)
*   **Answer**: Use `<vea><objN></vea>` for ONLY the direct answer objects (selective answer marking)
*   **Same Objects**: `<ver><obj1></ver>` and `<vea><obj1></vea>` refer to the same object, just used in different contexts
*   **No Other Tags**: Do NOT use any other tags in your output. Do NOT use <ver> in the `answer_caption` and do NOT use <vea> in the `reasoning`.

---

### **Output Format**

Your final output must be a single JSON object.

**A) If you generate a suitable question (score ≥ 7):**

```json
{
  "candidates": [
    {
      "id": 1,
      "question": "The question you formulated.",
      "reasoning": "Your step-by-step logical explanation, referencing objects like <ver><obj0></ver> and <ver><obj1></ver>.",
      "answer_objects": ["<objN>", "..."],
      "answer_caption": "Text answer connecting the objects. All objects (\"<vea><objN></vea>\", ..,) should be in the text.",
      "score": 8
    }
  ],
  "comments": "The description matches the image. The question requires complex, multi-object reasoning and has a clear, well-supported answer."
}
```

**B) If no suitable question can be formed (score < 7) or the description is inaccurate:**

```json
{
  "candidates": [],
  "comments": "A suitable question could not be generated because [your reason here, e.g., the objects lack meaningful relationships, the reasoning is too simple, or the description does not fit the image]."
}
```

---

### **Example**

**Note:** The following example is based on the hypothetical descriptions provided earlier. In a real task, you would be viewing the corresponding image.

```json
{
  "candidates": [
    {
      "id": 1,
      "question": "Which object's state suggests a different season than the others?",
      "reasoning": "To determine the likely season, we synthesize clues from multiple objects. The tree (<ver><obj2></ver>) has reddish-brown leaves, and the sky (<ver><obj1></ver>) is a clear, vibrant blue, both of which are strong indicators of a crisp autumn day. However, the grass (<ver><obj4></ver>) is lush and green with a few scattered yellow flowers, a condition more typical of spring or late summer. Therefore, the grass (<ver><obj4></ver>) presents evidence that contradicts the autumn season suggested by the other objects.",
      "answer_objects": ["<obj4>"], # This is the object that suggests a different season. No tag here.
      "answer_caption": "The grass (<vea><obj4></vea>) suggests a different season than the other objects.",
      "score": 9
    }
  ],
  "comments": "The question requires identifying a primary trend (the season) from multiple objects and then performing a secondary reasoning step to find a contradiction. This involves high-level synthesis and evaluation."
}
```
---

### **Input**

Descriptions: 
```text
{DESCRIPTIONS}
```
Please generate the output based on the descriptions and the provided image.
