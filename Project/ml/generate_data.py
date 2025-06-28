import requests
import pandas as pd
import random
import time
import os
from typing import List, Dict
import json
from dotenv import load_dotenv
load_dotenv()

class IntentDatasetGenerator:
    def __init__(self, api_key: str = None):

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("Please provide Groq API key either as parameter or set GROQ_API_KEY environment variable")
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"
        self.dataset = []
        self.generated_texts = set()
        
        # Define prompts for each intent class
        self.intent_prompts = {
            'email_send': {
                'description': 'Generate natural sentences for sending emails to people about various topics',
                'examples': [
                    "Send an email to John about the meeting tomorrow",
                    "Can you email Sarah the project update?",
                    "I need to send Mike an email regarding the budget",
                    "Email the team about the new policy changes",
                    "Please send an email to our client about the proposal"
                ],
                'context': 'Focus on various email scenarios including work emails, personal emails, forwarding information, sending updates, and requesting information via email.'
            },
            
            'calendar_schedule': {
                'description': 'Generate natural sentences for scheduling meetings, appointments, and calendar events',
                'examples': [
                    "Schedule a meeting with the team for tomorrow at 3pm",
                    "Can you book a conference room for Friday afternoon?",
                    "I need to schedule a call with the client next week",
                    "Please set up a meeting about the project review",
                    "Book a lunch meeting with Sarah for Thursday"
                ],
                'context': 'Include various types of meetings (team meetings, client calls, one-on-ones, interviews), different time references, and various meeting purposes.'
            },
            
            'web_search': {
                'description': 'Generate natural sentences for searching information online',
                'examples': [
                    "What's the weather like in New York today?",
                    "Search for the best restaurants near me",
                    "Can you find the latest news about the stock market?",
                    "Look up the population of Tokyo",
                    "What are the store hours for Target?"
                ],
                'context': 'Cover various search queries including weather, directions, business information, news, facts, prices, reviews, and general information lookup.'
            },
            
            'knowledge_query': {
                'description': 'Generate natural sentences for asking about company policies, procedures, and internal information',
                'examples': [
                    "What is our company's vacation policy?",
                    "Can you explain the expense reimbursement process?",
                    "What are the guidelines for remote work?",
                    "How does our performance review system work?",
                    "What's included in our health insurance plan?"
                ],
                'context': 'Focus on internal company knowledge, HR policies, procedures, benefits, guidelines, and organizational information.'
            },
            
            'general_chat': {
                'description': 'Generate natural sentences for casual conversation and social interaction',
                'examples': [
                    "How are you doing today?",
                    "What did you do over the weekend?",
                    "How's your day going so far?",
                    "What are your plans for the evening?",
                    "How was your vacation?"
                ],
                'context': 'Include greetings, casual questions, personal check-ins, small talk, and friendly conversation starters.'
            }
        }
    
    def call_groq_api(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.8) -> str:
      
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
    
    def generate_examples_for_intent(self, intent: str, num_examples: int = 200) -> List[str]:
      
        if intent not in self.intent_prompts:
            raise ValueError(f"Unknown intent: {intent}")
        
        prompt_info = self.intent_prompts[intent]
        examples = []
        batch_size = 25  # Generate examples in batches
        
        print(f"Generating {num_examples} examples for '{intent}'...")
        
        while len(examples) < num_examples:
            remaining = min(batch_size, num_examples - len(examples))
            
            prompt = f"""Generate {remaining} unique, natural, and grammatically correct sentences for the intent class '{intent}'.

Description: {prompt_info['description']}

Context: {prompt_info['context']}

Example sentences:
{chr(10).join(f"- {ex}" for ex in prompt_info['examples'])}

Requirements:
1. Each sentence should be unique and natural
2. Use proper grammar and sentence structure
3. Vary the sentence patterns and vocabulary
4. Make them realistic and conversational
5. Each sentence should clearly represent the '{intent}' intent
6. Return only the sentences, one per line, without bullets or numbering
7. Do not include any explanations or additional text

Generate {remaining} sentences:"""

            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that generates high-quality training data for natural language processing tasks. Generate diverse, natural, and grammatically correct sentences. Return only the requested sentences, one per line, with no additional formatting or explanations."},
                    {"role": "user", "content": prompt}
                ]
                
                generated_text = self.call_groq_api(messages, max_tokens=1200, temperature=0.8)
                batch_examples = [line.strip() for line in generated_text.split('\n') if line.strip()]
                
                # Filter out duplicates and add to examples
                for example in batch_examples:
                    # Clean up the example (remove bullets, numbers, etc.)
                    cleaned_example = example.lstrip('- •*1234567890. ').strip()
                    if cleaned_example and cleaned_example not in self.generated_texts and len(cleaned_example) > 10:
                        examples.append(cleaned_example)
                        self.generated_texts.add(cleaned_example)
                
                print(f"  Generated {len(examples)}/{num_examples} examples for '{intent}'")
                
                # Add a small delay to respect API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error generating examples for {intent}: {e}")
                time.sleep(2)  # Wait longer if there's an error
                continue
        
        return examples[:num_examples]
    
    def generate_complete_dataset(self, examples_per_intent: int = 200) -> pd.DataFrame:
        """
        Generate the complete dataset for all intent classes
        
        Args:
            examples_per_intent: Number of examples to generate per intent
            
        Returns:
            DataFrame with 'text' and 'intent' columns
        """
        print("🚀 Starting dataset generation with Groq API (Llama-3.3-70b-versatile)...")
        print(f"📊 Target: {examples_per_intent} examples per intent")
        print(f"🏷️  Intents: {list(self.intent_prompts.keys())}")
        print("=" * 60)
        
        for intent in self.intent_prompts.keys():
            examples = self.generate_examples_for_intent(intent, examples_per_intent)
            
            for example in examples:
                self.dataset.append({
                    'text': example,
                    'intent': intent
                })
            
            print(f"✅ Completed '{intent}': {len(examples)} examples")
            print("-" * 40)
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(self.dataset)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'intent_classification_dataset.csv'):
        """
        Save the dataset to CSV file
        
        Args:
            df: DataFrame containing the dataset
            filename: Output filename
        """
        df.to_csv(filename, index=False)
        
        print(f"\n✅ Dataset successfully saved to '{filename}'")
        print(f"📊 Total examples generated: {len(df)}")
        print(f"🏷️  Intent distribution:")
        intent_counts = df['intent'].value_counts().sort_index()
        for intent, count in intent_counts.items():
            print(f"   {intent}: {count} examples")
        
        print(f"\n👀 Sample examples from the dataset:")
        print("=" * 80)
        
        # Show 2 examples from each intent
        for intent in df['intent'].unique():
            intent_examples = df[df['intent'] == intent].sample(n=2, random_state=42)
            print(f"\n📝 {intent.upper()}:")
            for _, row in intent_examples.iterrows():
                print(f"   • {row['text']}")
        
        print("=" * 80)
    
    def generate_and_save(self, examples_per_intent: int = 200, filename: str = 'intent_classification_dataset.csv'):
        """
        Complete pipeline: generate dataset and save to file
        
        Args:
            examples_per_intent: Number of examples per intent class
            filename: Output CSV filename
        """
        df = self.generate_complete_dataset(examples_per_intent)
        self.save_dataset(df, filename)
        return df

def main():
    """
    Main function to run the dataset generation
    """
   
    try:
        generator = IntentDatasetGenerator()
        
        # Generate the dataset
        dataset = generator.generate_and_save(
            examples_per_intent=200,
            filename='intent_classification_dataset.csv'
        )
        
        print("\n🎉 Dataset generation completed successfully!")
        print(f"📁 Check 'intent_classification_dataset.csv' for the complete dataset")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 To fix this:")
        print("1. Get a free Groq API key from https://console.groq.com/keys")
        print("2. Set it as environment variable: export GROQ_API_KEY='your-groq-api-key'")
        print("3. Or pass it directly: generator = IntentDatasetGenerator(api_key='your-groq-api-key')")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()