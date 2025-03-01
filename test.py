import os
from openai import OpenAI

XAI_API_KEY = 'xai-YZWQumTiG1kNeYvCOqJFtWS9HCfHHxpsoEqMRLi9Yv3VXy3iMaV2q4TQFyQxouGcnOavfLleFFZJmPr8'
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Initialize conversation history
messages = [
    {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."}
]


def chat_with_grok():
    print("Welcome to Grok AI Chat! (Type 'quit' to exit)")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! Thanks for chatting!")
            break

        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        try:
            # Get response from Grok
            completion = client.chat.completions.create(
                model="grok-2-vision-latest",
                messages=messages
            )

            # Extract and print response
            response = completion.choices[0].message.content
            print("\nGrok:", response)

            # Add assistant's response to conversation history
            messages.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\nError: {str(e)}")
            messages.pop()  # Remove the last user message if there was an error


if __name__ == "__main__":
    chat_with_grok()
