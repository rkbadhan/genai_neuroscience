import os
import gradio as gr
from app.mcts import SimpleMCTS
from app.storage import process_query, handle_feedback

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

        def submit_query(query):
            response, query_id = process_query(query)
            return response, query_id

        submit_btn.click(
            submit_query,
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=port)
