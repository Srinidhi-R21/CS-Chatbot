# Intelligent Customer Support Chatbot
# This template provides a framework for building an automated customer support system

import os
import re
import json
import random
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# For NLP processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()]
)
logger = logging.getLogger("CustomerSupportBot")

class KnowledgeBase:
    """Manages the knowledge base for the chatbot"""
    
    def __init__(self, knowledge_file: str = "knowledge_base.json"):
        """Initialize the knowledge base from a JSON file"""
        self.knowledge_file = knowledge_file
        self.faq_data = self._load_knowledge()
        self.questions = [item["question"] for item in self.faq_data]
        self.answers = [item["answer"] for item in self.faq_data]
        self.categories = self._extract_categories()
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize_and_lemmatize)
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        
    def _load_knowledge(self) -> List[Dict]:
        """Load knowledge base from file or create default if not exists"""
        try:
            with open(self.knowledge_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Create a default knowledge base
            default_kb = [
                {
                    "question": "What are your business hours?",
                    "answer": "Our customer support is available 24/7. Our physical stores are open from 9 AM to 8 PM Monday through Saturday, and 10 AM to 6 PM on Sundays.",
                    "category": "general"
                },
                {
                    "question": "How do I reset my password?",
                    "answer": "You can reset your password by clicking the 'Forgot Password' link on the login page. We'll send a password reset link to your registered email address.",
                    "category": "account"
                },
                {
                    "question": "What is your return policy?",
                    "answer": "We offer a 30-day return policy for most items. Products must be in original condition with all packaging. Some restrictions apply for hygiene products.",
                    "category": "policy"
                }
            ]
            
            # Save the default knowledge base
            with open(self.knowledge_file, 'w') as f:
                json.dump(default_kb, f, indent=2)
                
            return default_kb
    
    def _extract_categories(self) -> List[str]:
        """Extract unique categories from the knowledge base"""
        return list(set(item.get("category", "general") for item in self.faq_data))
    
    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        return [self.lemmatizer.lemmatize(word.lower()) 
                for word in word_tokenize(text) 
                if word.isalnum()]
    
    def find_best_match(self, user_query: str, threshold: float = 0.6) -> Tuple[str, float]:
        """Find the best matching answer for a user query"""
        query_vector = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        best_match_index = similarities.argmax()
        best_match_score = similarities[best_match_index]
        
        if best_match_score >= threshold:
            return self.answers[best_match_index], best_match_score
        else:
            return None, best_match_score
    
    def add_entry(self, question: str, answer: str, category: str = "general") -> bool:
        """Add a new entry to the knowledge base"""
        try:
            self.faq_data.append({
                "question": question,
                "answer": answer,
                "category": category
            })
            
            # Update the vectors
            self.questions.append(question)
            self.answers.append(answer)
            if category not in self.categories:
                self.categories.append(category)
                
            # Recompute question vectors
            self.question_vectors = self.vectorizer.fit_transform(self.questions)
            
            # Save to file
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.faq_data, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error adding entry: {e}")
            return False

class CustomerSupportBot:
    """Main chatbot class that handles conversation flow"""
    
    def __init__(self):
        """Initialize the chatbot with knowledge base and conversation history"""
        self.kb = KnowledgeBase()
        self.conversation_history = []
        self.fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I don't have information on that. Would you like to speak to a human agent?",
            "I'm still learning and don't have an answer for that. Can I help with something else?",
            "That's beyond my current knowledge. Would you like me to create a support ticket?"
        ]
    
    def get_response(self, user_input: str) -> str:
        """Generate a response to user input"""
        # Record the conversation
        self.conversation_history.append({"user": user_input, "timestamp": datetime.now().isoformat()})
        
        # Check for special commands
        if user_input.lower() in ["quit", "exit", "bye"]:
            return "Thank you for chatting with us. Have a great day!"
        
        if "human" in user_input.lower() or "agent" in user_input.lower():
            return "I'll connect you with a human agent shortly. Please hold."
        
        # Process the query to find an answer
        answer, confidence = self.kb.find_best_match(user_input)
        
        if answer:
            response = answer
            logger.info(f"Found answer with confidence: {confidence:.2f}")
        else:
            # Use fallback responses if no good match
            response = random.choice(self.fallback_responses)
            logger.warning(f"No good match found. Confidence: {confidence:.2f}")
        
        # Record the bot's response
        self.conversation_history.append({"bot": response, "timestamp": datetime.now().isoformat()})
        
        return response
    
    def train_from_conversations(self, min_occurrences: int = 3) -> None:
        """Analyze conversation history to find common questions without answers"""
        if len(self.conversation_history) < 10:
            return
            
        # Find patterns of questions that led to fallback responses
        unanswered_questions = []
        
        for i in range(len(self.conversation_history) - 1):
            entry = self.conversation_history[i]
            if "user" in entry and "bot" in self.conversation_history[i+1]:
                user_msg = entry["user"]
                bot_msg = self.conversation_history[i+1]["bot"]
                
                if any(fallback in bot_msg for fallback in self.fallback_responses):
                    unanswered_questions.append(user_msg)
        
        # Count occurrences of similar questions
        # (This is a simplified approach; in production, use clustering)
        question_count = {}
        for q in unanswered_questions:
            normalized_q = " ".join(self.kb._tokenize_and_lemmatize(q))
            if normalized_q in question_count:
                question_count[normalized_q] += 1
            else:
                question_count[normalized_q] = 1
        
        # Report frequently asked but unanswered questions
        frequent_questions = []
        for q, count in question_count.items():
            if count >= min_occurrences:
                frequent_questions.append((q, count))
        
        if frequent_questions:
            logger.info(f"Found {len(frequent_questions)} frequent unanswered questions:")
            for q, count in frequent_questions:
                logger.info(f"Question: '{q}' - {count} occurrences")
            
            # In a real implementation, these could be sent to admins for review

    def save_conversation(self, user_id: str) -> bool:
        """Save the current conversation history to a file"""
        try:
            filename = f"conversation_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

# Example usage
if __name__ == "__main__":
    try:
        bot = CustomerSupportBot()
        # Add any additional initialization code here
    except Exception as e:
        logger.critical(f"Application error: {e}")
        raise
