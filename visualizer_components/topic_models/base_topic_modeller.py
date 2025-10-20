class TopicModeler:
    def __init__(self, chat_client):
        self.chat_client = chat_client
        self.topic_modeling_pairs = []

    def initialize_topic_modeling(self):
        self.chat_client.chat_with_groq(
            '''You are a precise topic modeling assistant. Follow these rules exactly:

1. Process text chunks sequentially as they arrive
2. Match chunks to existing topics using 70+ percent semantic similarity
3. Create new topics sequentially (0, 1, 2, ...) when no match found
4. Use underscore-separated keywords: keyword1_keyword2_keyword3
5. Output format: (topic_number, "keywords") - ALWAYS exactly this format
6. Maintain consistent topic-number-to-keywords mapping throughout
7. For existing topics: (number, "original_keywords")
8. For new topics: (next_number, "keyword1_keyword2_keyword3")
9. Respond with ONLY ONE pair per response, no other text
10. NEVER insert newlines, spaces, or any additional formatting
11. Output must be a single continuous line
12. Wait for user to provide chunks


Example: (0, "machine_learning_artificial_intelligence")

answer this prompt with Waiting for text chunks...''',
            model='llama-3.3-70b-versatile'
        )

    def process_chunks_for_topics(self, chunks):
        for chunk in chunks:
            string_result = self.chat_client.chat_with_groq(
                f'{chunk}',
                model='llama-3.3-70b-versatile'
            )
            self.topic_modeling_pairs.append(string_result)
        return self.topic_modeling_pairs

    def extract_topic_numbers_and_keywords(self):
        topic_numbers = []
        topic_keywords = []
        for s in self.topic_modeling_pairs:
            # Remove parentheses and split
            cleaned = s.strip('()')
            topic_id = int(cleaned.split(',')[0])
            topic_numbers.append(topic_id)
            topic_keywordss = cleaned.split(',')[1].strip().strip('"')
            topic_keywords.append(topic_keywordss)

        return topic_numbers, topic_keywords