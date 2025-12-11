from typing import Dict, Any, List, Optional
from openai import OpenAI
from ..services.rag import get_rag_service
from ..services.qdrant import get_qdrant_service
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentSkills:
    """
    Class containing reusable agent skills for the RAG chatbot
    These skills can be called by the OpenAI Assistant or used in the RAG service
    """

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag_service = get_rag_service()
        self.qdrant_service = get_qdrant_service()

    def explain_concept(self, concept: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Skill to explain a robotics/AI concept in detail
        """
        try:
            # Use RAG to find relevant information about the concept
            context_chunks = self.rag_service.retrieve_relevant_chunks(concept, limit=5)

            if not context_chunks:
                # If no specific content found, generate a general explanation
                prompt = f"""
                Explain the robotics/AI concept '{concept}' in detail.
                {f"User context: {user_context}" if user_context else ""}

                Provide a comprehensive explanation including:
                - Definition
                - Key principles
                - Applications in humanoid robotics
                - Technical details appropriate to the user's level
                """

                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in Physical AI and Humanoid Robotics. Provide detailed, accurate explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )

                return response.choices[0].message.content
            else:
                # Use retrieved context to explain the concept
                context = "\n\n".join([chunk["content"] for chunk in context_chunks])

                prompt = f"""
                Explain the robotics/AI concept '{concept}' in detail using the provided context.
                {f"User context: {user_context}" if user_context else ""}

                Context:
                {context}

                Provide a comprehensive explanation including:
                - Definition based on the context
                - Key principles
                - Applications in humanoid robotics
                - Technical details appropriate to the user's level
                - Citations to the source material
                """

                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in Physical AI and Humanoid Robotics. Provide detailed, accurate explanations based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )

                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error explaining concept {concept}: {str(e)}")
            return f"Sorry, I encountered an error while explaining '{concept}'. Please try rephrasing your question."

    def generate_quiz(self, topic: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Skill to generate a quiz about a specific topic
        """
        try:
            # Use RAG to find relevant information about the topic
            context_chunks = self.rag_service.retrieve_relevant_chunks(topic, limit=3)

            context = ""
            if context_chunks:
                context = "\n\n".join([chunk["content"] for chunk in context_chunks])

            # Prepare prompt for quiz generation
            prompt = f"""
            Generate a quiz about '{topic}' based on the provided context.
            {f"User context: {user_context}" if user_context else ""}

            Context:
            {context}

            Generate 5 multiple-choice questions with:
            - A clear question
            - 4 answer options (A, B, C, D)
            - The correct answer
            - A brief explanation of why the answer is correct

            Format the response as JSON with the following structure:
            {{
                "topic": "{topic}",
                "questions": [
                    {{
                        "question": "Question text",
                        "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                        "correct_answer": "A",
                        "explanation": "Explanation why this is correct"
                    }}
                ]
            }}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Physical AI and Humanoid Robotics education. Generate educational quizzes that test understanding of key concepts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            import json
            quiz_data = json.loads(response.choices[0].message.content)
            return quiz_data

        except Exception as e:
            logger.error(f"Error generating quiz for topic {topic}: {str(e)}")
            return {
                "topic": topic,
                "error": "Sorry, I encountered an error while generating the quiz. Please try another topic."
            }

    def translate_to_urdu(self, text: str, context: str = "general") -> str:
        """
        Skill to translate text to Urdu with appropriate technical terminology
        """
        try:
            prompt = f"""
            Translate the following text to Urdu. Pay special attention to technical terminology related to {context}.
            If technical terms don't have direct Urdu equivalents, provide the English term in parentheses after the Urdu translation.

            Text to translate:
            {text}

            Urdu translation:
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in technical content. Translate accurately while preserving technical meaning. Use proper Urdu script and appropriate technical terminology."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for more consistent translations
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error translating to Urdu: {str(e)}")
            return "Sorry, I encountered an error while translating to Urdu."

    def translate_to_roman_urdu(self, text: str, context: str = "general") -> str:
        """
        Skill to translate text to Roman Urdu (Urdu in Latin script) with appropriate technical terminology
        """
        try:
            prompt = f"""
            Translate the following text to Roman Urdu (Urdu written in Latin script). Pay special attention to technical terminology related to {context}.
            If technical terms don't have direct Roman Urdu equivalents, provide the English term after the Roman Urdu translation.

            Text to translate:
            {text}

            Roman Urdu translation:
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in Roman Urdu (Urdu written in Latin script). Translate accurately while preserving technical meaning. Use proper Romanization of Urdu words and appropriate technical terminology."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for more consistent translations
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error translating to Roman Urdu: {str(e)}")
            return "Sorry, I encountered an error while translating to Roman Urdu."

    def simplify_for_beginner(self, content: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Skill to simplify complex content for beginners
        """
        try:
            prompt = f"""
            Simplify the following content for a beginner in Physical AI and Humanoid Robotics.
            {f"User context: {user_context}" if user_context else ""}

            Content to simplify:
            {content}

            Provide a simplified explanation that:
            - Uses basic terminology
            - Breaks down complex concepts
            - Provides analogies where helpful
            - Maintains accuracy while improving accessibility
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert educator who specializes in making complex robotics and AI concepts accessible to beginners. Explain complex topics in simple terms without losing accuracy."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error simplifying content: {str(e)}")
            return "Sorry, I encountered an error while simplifying the content."

    def add_advanced_robotics_code(self, description: str) -> str:
        """
        Skill to generate advanced robotics code based on description
        """
        try:
            prompt = f"""
            Generate advanced robotics code for the following description:
            {description}

            Provide code examples in Python using common robotics libraries like:
            - ROS/ROS2
            - OpenCV
            - NumPy
            - PyTorch/TensorFlow
            - Gazebo/PyBullet simulation

            Include:
            - Proper comments explaining the code
            - Error handling
            - Best practices for robotics applications
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in advanced robotics programming. Generate clean, well-documented code that follows best practices for robotics applications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating robotics code: {str(e)}")
            return "Sorry, I encountered an error while generating the robotics code."

    def explain_diagram_with_vision(self, diagram_description: str) -> str:
        """
        Skill to explain diagrams or visual content (placeholder for when vision API is integrated)
        """
        try:
            # This would use OpenAI's vision API when integrated
            prompt = f"""
            Explain the following diagram or visual content in the context of Physical AI and Humanoid Robotics:
            {diagram_description}

            Provide a detailed explanation of:
            - What the diagram shows
            - How it relates to humanoid robotics concepts
            - Key components or processes illustrated
            - Implications for robot design or behavior
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Physical AI and Humanoid Robotics. Explain diagrams and visual content in detail, connecting them to robotics concepts and applications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error explaining diagram: {str(e)}")
            return "Sorry, I encountered an error while explaining the diagram."

    def retrieve_and_answer_from_selection(self, selected_text: str, question: str) -> str:
        """
        Skill to answer questions based specifically on selected text
        """
        try:
            # This skill specifically uses the selected text as context
            prompt = f"""
            Answer the following question based ONLY on the provided selected text:

            Selected text: {selected_text}

            Question: {question}

            Provide an accurate answer based only on the information in the selected text.
            If the answer cannot be determined from the selected text, clearly state that.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert assistant that answers questions based ONLY on the provided text. Do not use external knowledge. If the answer is not in the provided text, clearly state that."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error retrieving and answering from selection: {str(e)}")
            return "Sorry, I encountered an error while processing your question about the selected text."

    def generate_ros2_node(self, functionality: str) -> str:
        """
        Skill to generate a ROS2 node for specific functionality
        """
        try:
            prompt = f"""
            Generate a complete ROS2 node in Python for the following functionality:
            {functionality}

            Include:
            - Proper ROS2 node structure
            - Required imports
            - Node class with appropriate publishers/subscribers
            - Error handling
            - Comments explaining key sections
            - Example usage
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert ROS2 developer. Generate complete, functional ROS2 nodes in Python that follow ROS2 best practices and conventions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating ROS2 node: {str(e)}")
            return "Sorry, I encountered an error while generating the ROS2 node."

    def generate_urdf_from_description(self, robot_description: str) -> str:
        """
        Skill to generate URDF (Unified Robot Description Format) from robot description
        """
        try:
            prompt = f"""
            Generate a URDF (Unified Robot Description Format) file for a robot based on the following description:
            {robot_description}

            Include:
            - Proper URDF structure with links and joints
            - Appropriate physical properties (mass, inertia)
            - Visual and collision properties
            - Material definitions
            - Comments explaining key sections
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in robot modeling. Generate valid URDF files that properly describe robot geometry, kinematics, and physical properties."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating URDF: {str(e)}")
            return "Sorry, I encountered an error while generating the URDF."

    def debug_gazebo_launch(self, launch_file_content: str, error_description: str) -> str:
        """
        Skill to debug Gazebo launch files
        """
        try:
            prompt = f"""
            Debug the following Gazebo launch file and identify issues based on the error description:

            Launch file content:
            {launch_file_content}

            Error description:
            {error_description}

            Provide:
            - Identification of specific issues in the launch file
            - Corrected version of problematic sections
            - Explanation of why the issues occurred
            - Best practices for Gazebo launch files
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in Gazebo simulation and ROS launch files. Identify issues in launch files and provide corrected solutions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error debugging Gazebo launch: {str(e)}")
            return "Sorry, I encountered an error while debugging the Gazebo launch file."

# Global instance of AgentSkills
agent_skills = AgentSkills()

def get_agent_skills() -> AgentSkills:
    """Get the global instance of AgentSkills"""
    return agent_skills