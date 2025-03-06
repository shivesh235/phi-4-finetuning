import streamlit as st
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

class LocalMultimodalAIAssistant:
    def __init__(self, model_path="microsoft/Phi-4-multimodal-instruct"):
        """
        Initialize the multimodal AI assistant with Phi-4 model.
        
        Args:
            model_path (str): Hugging Face model identifier
        """
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).cuda()
        
        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # Prompt structure
        self.user_prompt = " "
        self.assistant_prompt = " "
        self.prompt_suffix = " "
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize uploaded image in session state
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
    
    def generate_response(self, prompt, image=None):
        """
        Generate a response from the model.
        
        Args:
            prompt (str): User's input message
            image (PIL.Image, optional): Uploaded image
        
        Returns:
            str: AI's generated response
        """
        # Prepare full context with chat history
        full_context = " ".join([
            f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg}" 
            for i, msg in enumerate(st.session_state.chat_history + [prompt])
        ])
        
        # Construct prompt with specific Phi-4 format
        full_prompt = f"{self.user_prompt}{full_context}{self.prompt_suffix}{self.assistant_prompt}"
        
        # Prepare inputs (with or without image)
        if image:
            full_prompt = f"{self.user_prompt}{full_prompt}"
            inputs = self.processor(
                text=full_prompt, 
                images=image, 
                return_tensors='pt'
            ).to('cuda:0')
        else:
            inputs = self.processor(
                text=full_prompt, 
                return_tensors='pt'
            ).to('cuda:0')
        
        # Generate response
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
        
        # Decode response
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
    
    def run(self):
        """
        Run the Streamlit app interface
        """
        # Page configuration
        st.set_page_config(page_title="Multimodal AI Assistant", page_icon="🤖")
        
        # App title
        st.title("🤖 Multimodal AI Assistant (Phi-4)")
        
        # Sidebar for image upload
        st.sidebar.header("Image Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image", 
            type=["jpg", "jpeg", "png"]
        )
        
        # Process uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Chat history display
        for msg in st.session_state.chat_history:
            if msg.startswith("Human:"):
                st.chat_message("human").write(msg.replace("Human: ", ""))
            else:
                st.chat_message("assistant").write(msg.replace("Assistant: ", ""))
        
        # Chat input
        if prompt := st.chat_input("Enter your message"):
            # Display user message
            st.chat_message("human").write(prompt)
            
            # Generate and display response
            with st.spinner("Generating response..."):
                # Use uploaded image if available
                response = self.generate_response(
                    prompt, 
                    image=st.session_state.uploaded_image
                )
                st.chat_message("assistant").write(response)
            
            # Update chat history
            st.session_state.chat_history.extend([
                f"Human: {prompt}",
                f"Assistant: {response}"
            ])
        
        # Clear chat history and image
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.experimental_rerun()

def main():
    assistant = LocalMultimodalAIAssistant()
    assistant.run()

if __name__ == "__main__":
    main()