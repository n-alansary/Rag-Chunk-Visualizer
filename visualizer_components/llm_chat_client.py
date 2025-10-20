from groq import Groq


class GroqChatClient:
    def __init__(self, api_key):
        self.groq_API_KEY = api_key
        self.client = Groq(api_key=self.groq_API_KEY)
        self.conversation_history = []

    def chat_with_groq(self, prompt, model='llama-3.3-70b-versatile'):
        try:
            self.conversation_history.append({"role": "user", "content": prompt})
            chat_completion = self.client.chat.completions.create(
                messages=self.conversation_history,
                model=model,
            )

            response = chat_completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response})

            print(f"{response}")
            return response
        except Exception as e:
            print(f"An error occurred: {e}")

