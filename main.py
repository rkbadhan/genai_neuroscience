import os
import json
import logging
import math
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, field
from PIL import Image
import io

import gradio as gr
import openai
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential
from tenacity.after import after_log
from tenacity.before import before_log
from tenacity.before_sleep import before_sleep_log
from fastapi import FastAPI
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
import os
# os.environ['OPENAI_API_KEY'] ="sk-Vy2IMkQku5bzY7zmbVT0Y1XZmsnOz5EanFqB3s5IgST3BlbkFJkHGcg7Dt1pJD1BWUIkx4Lh26Lwfwp5xlnm_hufF0AA"
os.environ['OPENAI_API_KEY'] ="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"
client = OpenAI()
model_name="gpt-4o"

# Configuration and Models
class Settings(BaseSettings):
    # model_name: str = model_name  # Fixed typo from "gpt-4o"
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
    def __init__(self, problem: str, image_path: Optional[str] = None, max_iterations: Optional[int] = None):
        self.problem = problem
        self.image_path = image_path
        self.max_iterations = max_iterations or settings.max_iterations
        self.client = client
        self.logger = logging.getLogger(__name__)

    def _call_openai(self, system_content: str, user_content: str, include_image: bool = False) -> str:
        try:
            if include_image and self.image_path:
                # Handle image-based analysis
                with open(self.image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                messages = [
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_data}"
                                }
                            }
                        ]
                    }
                ]
            else:
                # Handle text-only analysis
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _process_image(self) -> str:
        """Process the image and return analysis results"""
        if not self.image_path:
            return ""

        try:
            self.logger.info(f"Processing image: {self.image_path}")
            
            analysis = self._call_openai(
                "You are a medical imaging expert. Analyze this medical image comprehensively.",
                """Please provide a detailed analysis of this medical image including:
                1. Type of imaging (MRI, CT, X-ray, etc.)
                2. Anatomical orientation
                3. Key findings and abnormalities
                4. Clinical implications""",
                include_image=True
            )
            
            return f"\n\n## Image Analysis\n{analysis}"
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            return "\n\nImage processing failed. Proceeding with text analysis only."

    def _generate_initial_answer(self) -> Node:
        self.logger.info("Generating initial answer")
        image_analysis = self._process_image() if self.image_path else ""
        full_context = f"{self.problem}\n\n{image_analysis}"
        response = self._call_openai(INITIAL_RESPONSE_PROMPT, full_context)
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
                f"Problem: {self.problem}\nAnswer: {answer}"
            )
            return float(response.strip())
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
        self.logger.info("Starting MCTS")
        try:
            # Generate initial answer
            self.root = self._generate_initial_answer()
            initial_response = f"## Initial Assessment\n\n{self.root.answer}"
            yield initial_response + "\n\n*Processing improvements...*"
            
            current = self.root
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
                
                yield f"## Iteration {iteration + 1}/{self.max_iterations}\n\n{refined_answer}"

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
            yield f"## Final Assessment\n\n{best_node.answer}"
            
        except Exception as e:
            self.logger.error(f"Error during MCTS execution: {str(e)}")
            yield f"## Error\n\nError during processing: {str(e)}"

# Storage Functions
def save_query(query: str, response: str) -> str:
    timestamp = datetime.now().isoformat()
    query_data = {
        "timestamp": timestamp,
        "query": query,
        "response": response
    }
    
    os.makedirs("data/queries", exist_ok=True)
    
    filename = f"data/queries/{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(query_data, f, indent=2)
    
    return timestamp

def save_feedback(query_id: str, is_positive: bool, text_feedback: str = "") -> None:
    feedback_data = {
        "query_id": query_id,
        "is_positive": is_positive,
        "text_feedback": text_feedback,
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs("data/feedback", exist_ok=True)
    
    filename = f"data/feedback/{query_id}_feedback.json"
    with open(filename, "w") as f:
        json.dump(feedback_data, f, indent=2)

def process_query(query: str, image: Optional[str] = None):
    try:
        logger.info(f"Processing query with image: {bool(image)}")
        
        if image:
            try:
                img = Image.open(image)
                logger.info(f"Image format: {img.format}, Size: {img.size}")
                img.close()
            except Exception as e:
                logger.error(f"Error verifying image: {str(e)}")
                yield "## Error\n\nUnable to process the image. Please check the format.", ""
                return

        mcts = SimpleMCTS(problem=query, image_path=image)
        last_response = None
        
        for response in mcts.run():
            last_response = response
            yield response, ""
            
        query_id = save_query(query, last_response)
        yield last_response, query_id
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        yield "## Error\n\nAn error occurred while processing your query. Please try again.", ""

def handle_feedback(is_positive: bool, query_id: str, text_feedback: str) -> str:
    try:
        if query_id:
            save_feedback(query_id, is_positive, text_feedback)
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
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Describe the clinical case",
                    placeholder="Enter patient symptoms and clinical findings...",
                    lines=5
                )
                image_input = gr.Image(
                    label="Upload Medical Image (optional)",
                    type="filepath"
                )
                submit_btn = gr.Button("Get Consultation", variant="primary")
            
            with gr.Column(scale=1):
                output = gr.Markdown(label="Consultation Response")
                with gr.Row():
                    upvote_btn = gr.Button("👍 Helpful")
                    downvote_btn = gr.Button("👎 Not Helpful")
                
                text_feedback = gr.Textbox(
                    label="Additional Feedback",
                    placeholder="Please provide any additional comments or suggestions...",
                    lines=3
                )
                submit_feedback_btn = gr.Button("Submit Feedback")
                feedback_message = gr.Textbox(label="", interactive=False)

        current_query_id = gr.State("")

        submit_btn.click(
            process_query,
            inputs=[query_input, image_input],
            outputs=[output, current_query_id]
        )

        def submit_positive_feedback(query_id, feedback_text):
            result = handle_feedback(True, query_id, feedback_text)
            return result

        def submit_negative_feedback(query_id, feedback_text):
            result = handle_feedback(False, query_id, feedback_text)
            return result

        upvote_btn.click(
            submit_positive_feedback,
            inputs=[current_query_id, text_feedback],
            outputs=[feedback_message]
        )

        downvote_btn.click(
            submit_negative_feedback,
            inputs=[current_query_id, text_feedback],
            outputs=[feedback_message]
        )

        submit_feedback_btn.click(
            lambda id, text: handle_feedback(None, id, text),
            inputs=[current_query_id, text_feedback],
            outputs=[feedback_message]
        )

    return interface

# Create the interface
interface = create_interface()

# Mount the app
app = gr.mount_gradio_app(
    FastAPI(), 
    interface, 
    path="/"
)
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
