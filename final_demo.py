from llm_topic_modelling_structured import TopicModelingPipeline

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from dotenv import load_dotenv
import os

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")


# reader = PdfReader('harry_potter_book.pdf')
# sample_text = ""

# for page in reader.pages:
#     sample_text += page.extract_text() + '\n'

sample_text ='''
Machine Learning and Artificial Intelligence

Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention. The iterative aspect of
machine learning is important because as models are exposed to new data, they are able to
independently adapt. They learn from previous computations to produce reliable, repeatable decisions and results.

Deep learning is a subset of machine learning in artificial intelligence that has networks capable
of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural
learning or deep neural network. Deep learning models can achieve state-of-the-art accuracy,
sometimes exceeding human-level performance. Models are trained using large sets of labeled data
and neural network architectures containing many layers.

Climate Change and Environmental Science

Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may
be natural, such as through variations in the solar cycle. But since the 1800s, human activities
have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas.
Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around Earth,
trapping the sun's heat and raising temperatures.

The consequences of climate change now include intense droughts, water scarcity, severe fires,
rising sea levels, flooding, melting polar ice, catastrophic storms and declining biodiversity.
Scientists use observations from the ground, air and space, along with theoretical models, to monitor
and study past, present and future climate change.

Renewable energy is energy derived from natural sources that are replenished at a higher rate than
they are consumed. Sunlight and wind, for example, are such sources that are constantly being replenished.
Renewable energy sources are plentiful and all around us. Clean energy has far more to recommend it
than just being green. The renewable energy industry is growing rapidly across the globe.

Medical Research and Healthcare

Medical research involves research in a wide range of fields, such as biology, chemistry, pharmacology
and toxicology with the goal of developing new medicines or medical procedures to improve health and
well-being. Clinical trials are research studies performed on people aimed at evaluating a medical,
surgical, or behavioral intervention. They are the primary way researchers find out if a new treatment
is safe and effective in people.

Precision medicine is an emerging approach for disease treatment and prevention that takes into account
individual variability in genes, environment, and lifestyle for each person. This approach allows doctors
and researchers to predict more accurately which treatment and prevention strategies for a particular
disease will work in which groups of people. It is in contrast to a one-size-fits-all approach.

Gene therapy is a technique that modifies a person's genes to treat or cure disease. Gene therapies
can work by several mechanisms including replacing a disease-causing gene with a healthy copy of the gene,
inactivating a disease-causing gene that is not functioning properly, or introducing a new or modified
gene into the body to help treat a disease.

Space Exploration and Astronomy

Space exploration is the use of astronomy and space technology to explore outer space. While the exploration
of space is carried out mainly by astronomers with telescopes, its physical exploration is conducted both by
uncrewed robotic space probes and human spaceflight. Space exploration has often been used as a proxy
competition for geopolitical rivalries such as the Cold War.

The International Space Station is a modular space station in low Earth orbit. It is a multinational
collaborative project involving five participating space agencies. The ISS serves as a microgravity and
space environment research laboratory in which scientific research is conducted in astrobiology, astronomy,
meteorology, physics, and other fields.

Mars exploration is the study of Mars by spacecraft. Probes sent from Earth, beginning in the late 20th century,
have yielded a large increase in knowledge about the Martian system, focused primarily on understanding its
geology and habitability potential. Engineering interplanetary journeys is complicated and the exploration of
Mars has experienced a high failure rate.

Quantum Computing and Physics

Quantum computing is a type of computation that harnesses the collective properties of quantum states,
such as superposition, interference, and entanglement, to perform calculations. The devices that perform
quantum computations are known as quantum computers. Quantum computers are believed to be able to solve
certain computational problems substantially faster than classical computers.

Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact,
or share spatial proximity in a way such that the quantum state of each particle of the group cannot be
described independently of the state of the others, including when the particles are separated by a large distance.

Quantum supremacy is the goal of demonstrating that a programmable quantum device can solve a problem that
no classical computer can solve in any feasible amount of time. The term was coined by John Preskill in 2012,
originally requiring quantum computers to solve problems that are impossible for classical computers.

Economic Theory and Finance

Economics is a social science that studies the production, distribution, and consumption of goods and services.
Economics focuses on the behavior and interactions of economic agents and how economies work. Microeconomics
analyzes what's viewed as basic elements in the economy, including individual agents and markets, their
interactions, and the outcomes of interactions.

Cryptocurrency is a digital currency designed to work as a medium of exchange through a computer network
that is not reliant on any central authority, such as a government or bank, to uphold or maintain it.
Bitcoin, first released as open-source software in 2009, is the first decentralized cryptocurrency.

The stock market refers to public markets that exist for issuing, buying, and selling stocks that trade
on a stock exchange or over-the-counter. Stocks represent ownership in a publicly-traded company. The stock
market is one of the most important components of a free-market economy, as it provides companies with access
to capital in exchange for giving investors a slice of ownership. and create jobs.

Machine learning algorithms continue to evolve and improve, driving innovation across various industries.
The field is rapidly advancing with new techniques such as transfer learning and few-shot learning.
Artificial intelligence systems are increasingly being integrated into everyday applications, from virtual assistants to recommendation engines.
The future of AI holds promise for solving complex global challenges through advanced pattern recognition and predictive modeling.

'''



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    length_function=len,
    )

chunks = text_splitter.split_text(sample_text)

pipeline = TopicModelingPipeline(
    api_key=groq_api_key
    )
pipeline.run_pipeline( chunks[:150] , output_file="topic_visualization_3d.html")