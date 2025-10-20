import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP


def create_hover_text_of_chunks(text, max_len=100):
        """Break text into lines of max_len characters, splitting at last space"""
        if len(text) <= max_len:
            return text

        lines = []
        start = 0

        while start < len(text):
            # Get the next chunk
            end = start + max_len

            # If we're at the end, take the rest
            if end >= len(text):
                lines.append(text[start:])
                break

            # Find the last space within the chunk
            chunk = text[start:end]
            last_space = chunk.rfind(' ')

            if last_space != -1:  # Found a space
                lines.append(chunk[:last_space])
                start = start + last_space + 1  # Move past the space
            else:  # No space found, break at max_len
                lines.append(chunk)
                start = end

        return '<br>'.join(lines)


class TopicVisualizer:
    def __init__(self):
        pass

    def create_3d_visualization(self, chunks, topic_numbers, topic_keywords, 
                                dim_reduction_method = None,
                                output_file="topic_visualization_3d.html"):
        # Create a mapping from topic numbers to their keywords for labels
        topic_dict = {}
        for i, (topic_num, keyword) in enumerate(zip(topic_numbers, topic_keywords)):
            if topic_num not in topic_dict:
                topic_dict[topic_num] = keyword

        # Create TF-IDF vectors for your chunks
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunks)

        # Apply UMAP for 3D dimensionality reduction
        if dim_reduction_method is None:
            dim_reduction_method = UMAP(n_components=3, random_state=42, n_neighbors=10, min_dist=0.1)
        
        embeddings_3d = dim_reduction_method.fit_transform(tfidf_matrix)

        # Create hover text with topic information
        hover_text = [
            f"Chunk {i}<br>Topic: {topic_num}<br>Keywords: {topic_keywords[i]}<br>Text: <br>{create_hover_text_of_chunks(chunks[i], max_len=100)}"
            for i, (topic_num, _) in enumerate(zip(topic_numbers, topic_keywords))
        ]

        # Create 3D scatter plot with Plotly - using valid colorscale
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=topic_numbers,
                colorscale='viridis',  # Use a valid Plotly colorscale
                opacity=0.7,
                colorbar=dict(title="Topic")
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',  # Changed from <b>%{text}</b>
            name='Topics'
        )])

        fig.update_layout(
            title='3D Topic Visualization',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            width=800,
            height=600
        )

        # Save as HTML file
        fig.write_html(output_file)