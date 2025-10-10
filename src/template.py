from liquid import Template

knowledge_summary_template = '''Given the following Question and Document, please distillate useful information from the Document to address the Question effectively. 
Ensure that the useful information is directly supported by evidence found within the document, and avoid unnecessary details.
Focus on extracting all information relevant to understanding the Question and evaluating any of the answer options, rather than only information that supports a presumed correct answer.
Not provided information should not appear in your response.

If the document does not contain relevant information for answering the Question, return "No useful information".  
Only output the useful information directly, without any additional explanations or formatting.  
## Don’t provide the answer to the question. Instead, focus on distillate useful information that are essential for answering the question accurately.

******Example:******
{example}


******Real Test:******
# Retrieved Document
{documents}

# Question
{question}

# Useful information
'''


knowledge_explore_template = '''Given the Question and the several pieces of Useful Information extracted from retrieved documents, your task is to identify the most important missing knowledge required to answer the Question thoroughly.

Follow these steps carefully:
Step 1. Analyze the Question and decompose it into key knowledge points that are necessary for a complete answer.
Step 2. Compare these required knowledge points with the provided Useful Information and determine which knowledge points are missing or insufficient.
Step 3. Select the 3 most critical and diverse missing knowledge points that are not overlapping or redundant. Express each missing knowledge point as a concise conceptual title (around one short phrase or clause), not as a full explanatory sentence. The title should be specific enough to clearly indicate the missing medical knowledge (e.g., “Autosomal dominant inheritance pattern of NF2”) rather than overly generic or vague.

You should response with English and do not use any other languages.
Your output should follow the following format, do not output other additional content:
- Reasoning: [Detailed explanation]
- Knowledge 1: [Conceptual title 1]
- Knowledge 2: [Conceptual title 2]
- Knowledge 3: [Conceptual title 3]

******Example:******
{example}


******Real Test:******
# Useful Information
{documents}

# Question
{question}

# Output
'''


rag_template = '''You are a professional medical expert to answer the # Question using the # Retrieved Documents. Please first think step-by-step and then answer the question. Your responses will be used for research purposes only, so please have a definite answer.
When answering, you should first try to extract and utilize useful information from the retrieved documents. If the documents do not contain sufficient or relevant information to answer the question, then you should rely on your own medical knowledge to provide a complete and accurate response.
You should organize your output in a json formatted as Dict{{"reasoning": Str(explanation), "answer_choice": Str{{A/B/C/...}}}}. 

# Retrieved Documents
{documents}

# Question
{question}

You should organize your output in a json formatted as Dict{{"reasoning": Str(explanation), "answer_choice": Str{{A/B/C/...}}}}. 
'''

gen_template = '''You are a professional medical expert tasked with generating a helpful background document containing relevant medical knowledge that can assist in answering the given multiple-choice question.  

Follow these instructions carefully: 
- Generate the background document based only on the provided single knowledge point. Do not introduce other knowledge points.
- Provide supporting knowledge, explanations, or relevant context related to this knowledge point.    
- Do not include any guesses or speculation about the correct answer.
- Do not contain any phrases like “the answer is A” or guesses about which option is correct. 
- Be written in English and not exceed *256 words*.  
 
Just return the background document. 


### Question: 
{question}

### Knowledge Point:  
{content} 

### Background Document:
'''

gen_wo_know_template = '''You are a professional medical expert to generate a helpful background document containing relevant medical knowledge that can assist in answering the given multiple-choice question. 

Follow these instructions carefully:  
- Generate the background document based only on the information contained in the question and general medical knowledge.
- Provide supporting knowledge, explanations, or relevant context that are relevant to understanding the question.  
- Do not include any guesses or speculation about the correct answer.
- Do not contain any phrases like “the answer is A” or guesses about which option is correct. 
- Be written in English and not exceed *256 words*.  

Just return the background document. 


### Question: 
{question}

### Background Document:
'''

llm_selection_prompt = '''I will provide you with 10 passages, each indicated by a numerical identifier [id]. Your task is to select the top-5 most useful passages for answering the given medical question:
{question}

# Documents
{documents}

# Question
{question}

Please follow the steps carefully:
Step 1. Information Requirements Identification:
Please list up the key knowledge points to answer the question thoroughly.
Step 2. Requirement-to-Passage Mapping:
Map passages to each knowledge point in Step 1, group them accordingly, and put irrelevant ones into a "No Useful Information" group.
Step 3. Document Selection for Completeness and Conciseness
Select up to 5 passages that maximize coverage across key knowledge points while avoiding redundancy.

Your output should follow the following format, do not output other additional content:
- Reasoning: [Detailed explanation]
- Final Selection: [id1] [id2] [id3] [id4] [id5]
'''