from visualizer_components.llm_chat_client import GroqChatClient
from visualizer_components.topic_models.base_topic_modeller import TopicModeler
from visualizer_components.topic_visualizer import TopicVisualizer
from umap import UMAP




class TopicModelingPipeline:
    def __init__(self, api_key):
        self.chat_client = GroqChatClient(api_key)
        # self.text_processor = TextProcessor(chunk_size, chunk_overlap)
        self.topic_modeler = TopicModeler(self.chat_client)
        self.visualizer = TopicVisualizer()

    def run_pipeline(self  , chunks , output_file="topic_visualization_3d.html"):
        # Extract and process text
        # chunks = self.text_processor.split_text(sample_text)

        # Initialize and run topic modeling
        self.topic_modeler.initialize_topic_modeling()
        topic_pairs = self.topic_modeler.process_chunks_for_topics(chunks)
        topic_numbers, topic_keywords = self.topic_modeler.extract_topic_numbers_and_keywords()

        print(topic_pairs)
        print(topic_numbers)
        print(type(topic_numbers[0]))
        print(topic_keywords)
        print(type(topic_keywords[0]))

        # Create visualization
        self.visualizer.create_3d_visualization(
            chunks, 
            topic_numbers, 
            topic_keywords,
            # dim_reduction_method = UMAP(n_components=3, random_state=42, n_neighbors=10, min_dist=0.1),
            output_file=output_file
            
        )
