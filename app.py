from flask import Flask
# Cell 1: Package Installation
%%capture
!pip install gradio==4.12.0 openai==1.8.0 pydantic==2.5.2 python-dotenv==1.0.0 tenacity==8.2.3 pydantic-settings==2.1.0

# Cell 2: Imports and Basic Setup
import os
import json
from pathlib import Path
from typing import Optional, Any, List, Dict
import openai
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass, field
import math
import gradio as gr
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from tenacity.after import after_log
from tenacity.before import before_log
from tenacity.before_sleep import before_sleep_log

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
import getpass
OPENAI_API_KEY = getpass.getpass('Enter your OpenAI API key: ')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create necessary directories
!mkdir -p logs data/queries data/feedback

# Cell 3: Configuration and Models
class Settings(BaseSettings):
    openai_api_key: str = ""
    model_name: str = "gpt-4"
    max_iterations: int = 2
    temperature: float = 0.8
    max_tokens: int = 1024
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
settings.openai_api_key = OPENAI_API_KEY

class Eval(BaseModel):
    score: float

@dataclass
class Node:
    answer: str
    parent: Optional['Node'] = None
    children: list['Node'] = field(default_factory=list)
    visits: int = 0
    quality_score: float = 0

    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'visits': self.visits,
            'quality_score': self.quality_score,
            'children': [child.to_dict() for child in self.children]
        }

# Cell 4: Prompts
INITIAL_RESPONSE_PROMPT = '''Role: Senior Neurosurgeon and Neurophysician with dual board certification

Response Protocol:
1. Initial Clinical Diagnosis
   - Differential diagnoses prioritized by probability
   - Key neurological findings supporting each diagnosis

2. Radiological Confirmation
   - Specific imaging modalities required
   - Expected radiological findings
   - Critical areas to evaluate

3. Clinical Assessment
   - Essential neurological examination parameters
   - Required functional assessments
   - Immediate intervention indicators if applicable

Guidelines:
- Responses assume expert-level understanding
- Use standard neurosurgical terminology
- Focus on critical decision points
- Highlight any surgical urgency indicators'''

CRITIQUE_PROMPT = '''You are an expert Neurosurgical Response Evaluator with extensive clinical and academic experience.

Evaluation Criteria:
1. Diagnostic Precision
   - Accuracy of differential diagnosis ranking
   - Appropriateness of radiological recommendations
   - Validity of clinical correlations

2. Clinical Relevance
   - Emergency recognition
   - Surgical vs. conservative management justification
   - Risk assessment accuracy

3. Documentation Alignment
   - Evidence-based recommendations
   - Protocol adherence'''

# Cell 5: MCTS Implementation
class SimpleMCTS:
    def __init__(self, problem: str, max_iterations: Optional[int] = None):
        self.problem = problem
        self.max_iterations = max_iterations or settings.max_iterations
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.root: Optional[Node] = None
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def _call_openai(
        self,
        system_content: str,
        user_content: str,
        response_format: Optional[Any] = None
    ) -> str:
        try:
            if response_format:
                response = self.client.beta.chat.completions.parse(
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    response_format=response_format,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _generate_initial_answer(self) -> Node:
        self.logger.info("Generating initial answer")
        response = self._call_openai(INITIAL_RESPONSE_PROMPT, self.problem)
        return Node(answer=response)

    def _get_critique(self, answer: str) -> str:
        self.logger.info("Getting critique for answer")
        return self._call_openai(
            CRITIQUE_PROMPT,
            f"Problem: {self.problem}\nAnswer: {answer}"
        )

    def _refine_answer(self, answer: str, critique: str) -> str:
        self.logger.info("Refining answer based on critique")
        return self._call_openai(
            "Improve the answer based on the critique.",
            f"Problem: {self.problem}\nCurrent Answer: {answer}\nCritique: {critique}"
        )

    def _evaluate_answer(self, answer: str) -> float:
        try:
            self.logger.info("Evaluating answer quality")
            response = self._call_openai(
                "Rate the answer quality from 0-100. Return only the number.",
                f"Problem: {self.problem}\nAnswer: {answer}",
                Eval
            )
            json_output = json.loads(response)
            return float(json_output['score'])
        except Exception as e:
            self.logger.error(f"Failed to evaluate answer: {str(e)}")
            return 0.0

    def _select_best_node(self, node: Node) -> Node:
        if not node.children:
            return node

        def uct_score(child: Node) -> float:
            exploration = math.sqrt(2 * math.log(node.visits) / (child.visits + 1))
            return child.quality_score + exploration

        return max(node.children, key=uct_score)

    def run(self) -> str:
        self.logger.info(f"Starting MCTS with max_iterations={self.max_iterations}")
        self.root = self._generate_initial_answer()
        current = self.root

        try:
            # First return the initial answer while processing continues
            initial_response = self.root.answer
            yield initial_response + "\n\nProcessing further improvements..."
            
            for iteration in range(self.max_iterations):
                self.logger.info(f"Starting iteration {iteration + 1}")
                
                while current.children:
                    current = self._select_best_node(current)

                critique = self._get_critique(current.answer)
                refined_answer = self._refine_answer(current.answer, critique)
                quality_score = self._evaluate_answer(refined_answer)

                new_node = Node(
                    answer=refined_answer,
                    parent=current,
                    quality_score=quality_score
                )
                current.children.append(new_node)

                # Yield progress updates
                yield f"Iteration {iteration + 1}/{self.max_iterations} complete...\n\n{refined_answer}"

                while current:
                    current.visits += 1
                    current.quality_score = max(
                        current.quality_score,
                        max((c.quality_score for c in current.children), default=0)
                    )
                    current = current.parent

                current = self.root

            best_node = max(
                [self.root] + [c for n in [self.root] for c in n.children],
                key=lambda x: x.quality_score
            )
            
            self.logger.info("MCTS completed successfully")
            yield best_node.answer
            
        except Exception as e:
            self.logger.error(f"Error during MCTS execution: {str(e)}")
            yield f"Error occurred: {str(e)}"

# Cell 6: Data Storage Functions
def save_query(query: str, response: str) -> str:
    """Save the query and response with timestamp"""
    timestamp = datetime.now().isoformat()
    query_data = {
        "timestamp": timestamp,
        "query": query,
        "response": response
    }
    
    filename = f"data/queries/{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(query_data, f, indent=2)
    
    return timestamp

def save_feedback(query_id: str, is_positive: bool) -> None:
    """Save user feedback"""
    feedback_data = {
        "query_id": query_id,
        "is_positive": is_positive,
        "timestamp": datetime.now().isoformat()
    }
    
    filename = f"data/feedback/{query_id}_feedback.json"
    with open(filename, "w") as f:
        json.dump(feedback_data, f, indent=2)

def process_query(query: str) -> tuple[str, str]:
    """Process the query using MCTS and return response with query ID"""
    try:
        mcts = SimpleMCTS(problem=query)
        response = mcts.run()
        query_id = save_query(query, response)
        return response, query_id
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return "An error occurred while processing your query. Please try again.", ""

def handle_feedback(is_positive: bool, query_id: str) -> str:
    """Handle user feedback"""
    try:
        if query_id:
            save_feedback(query_id, is_positive)
            return "Thank you for your feedback!"
        return "No query to provide feedback for."
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return "An error occurred while saving your feedback."



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, Kubernetes!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
