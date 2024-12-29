import os
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, field

import gradio as gr
import openai
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential
from tenacity.after import after_log
from tenacity.before import before_log
from tenacity.before_sleep import before_sleep_log

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration and Models
class Settings(BaseSettings):
    openai_api_key: str = "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"
    model_name: str = "gpt-4o"  # Fixed typo from "gpt-4o"
    max_iterations: int = 2
    temperature: float = 0.8
    max_tokens: int = 1024
    
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

settings = Settings()


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

# Prompts
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
   - Immediate intervention indicators if applicable'''

CRITIQUE_PROMPT = '''You are an expert Neurosurgical Response Evaluator with extensive clinical and academic experience.

Evaluation Criteria:
1. Diagnostic Precision
   - Accuracy of differential diagnosis ranking
   - Appropriateness of radiological recommendations
   - Validity of clinical correlations

2. Clinical Relevance
   - Emergency recognition
   - Surgical vs. conservative management justification
   - Risk assessment accuracy'''

# MCTS Implementation
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
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            if response_format:
                response = self.client.beta.chat.completions.parse(
                    model=settings.model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=settings.model_name,
                    messages=messages,
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

    def run(self):
        self.logger.info(f"Starting MCTS with max_iterations={self.max_iterations}")
        self.root = self._generate_initial_answer()
        current = self.root
        
        try:
            # Return initial answer
            initial_response = self.root.answer
            yield initial_response + "\n\nProcessing improvements..."
            
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
                
                yield f"Iteration {iteration + 1}/{self.max_iterations}...\n\n{refined_answer}"

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

# Storage Functions
def save_query(query: str, response: str) -> str:
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
    feedback_data = {
        "query_id": query_id,
        "is_positive": is_positive,
        "timestamp": datetime.now().isoformat()
    }
    
    filename = f"data/feedback/{query_id}_feedback.json"
    with open(filename, "w") as f:
        json.dump(feedback_data, f, indent=2)

def process_query(query: str):
    try:
        mcts = SimpleMCTS(problem=query)
        for response in mcts.run():
            yield response, ""  # Yield intermediate responses
        query_id = save_query(query, response)
        yield response, query_id  # Yield final response with query ID
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        yield "An error occurred while processing your query. Please try again.", ""

def handle_feedback(is_positive: bool, query_id: str) -> str:
    try:
        if query_id:
            save_feedback(query_id, is_positive)
            return "Thank you for your feedback!"
        return "No query to provide feedback for."
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return "An error occurred while saving your feedback."

# Gradio Interface
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Neurosurgical Consultation Assistant")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Describe the clinical case",
                    placeholder="Enter patient symptoms and clinical findings...",
                    lines=5
                )
                submit_btn = gr.Button("Get Consultation")
            
        with gr.Column():
            output = gr.Textbox(label="Consultation Response", lines=10)
            with gr.Row():
                upvote_btn = gr.Button("👍 Helpful")
                downvote_btn = gr.Button("👎 Not Helpful")

        current_query_id = gr.State("")

        submit_btn.click(
            process_query,
            inputs=[query_input],
            outputs=[output, current_query_id]
        )

        upvote_btn.click(
            lambda id: handle_feedback(True, id),
            inputs=[current_query_id],
            outputs=[]
        )

        downvote_btn.click(
            lambda id: handle_feedback(False, id),
            inputs=[current_query_id],
            outputs=[]
        )

    return interface

# Create the interface
# At the end of your file, modify the app exposure:
interface = create_interface()

# Instead of directly exposing the app property, wrap it properly:
app = gr.mount_gradio_app(
    FastAPI(), 
    interface, 
    path="/"
)

if __name__ == "__main__":
    # This block is only for local development
    port = int(os.environ.get("PORT", 8080))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        allowed_paths=["logs", "data"],
        show_error=True
    )
