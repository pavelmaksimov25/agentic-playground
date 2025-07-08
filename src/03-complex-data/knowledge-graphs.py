import os
import base64
from openai import OpenAI

from pydantic import BaseModel, Field
from typing import List
import graphviz
from IPython.display import Image
from dotenv import load_dotenv

load_dotenv()

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

class Node(BaseModel):
    id: int
    label: str
    attribute: str
    color: str

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    description: str = None

    def visualize(self):
        dot = graphviz.Digraph(comment="Knowledge Graph")
        dot.format = 'svg'

        for node in self.nodes:
            dot.node(str(node.id), label=node.label, color=node.color, style='filled', fillcolor=node.color)

        for edge in self.edges:
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)
        
        dot.render("social_network", view=False, cleanup=True)
    

def generate_graph(input) -> KnowledgeGraph:
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages = [{"role" : "assistant", "content" : f""" Help me understand the following by describing as a detailed knowledge graph:  {input}"""}],
        response_format = KnowledgeGraph)
    
    print(completion)

    return completion.choices[0].message.parsed

data = """
 What is the relationship between cars, wheels and trains in relation to the speed that can be achieved relative to the friction of the wheel and the difference of the material the wheel is made of comparing rubber and steel wheels."""

graph = generate_graph(data + " Please generate a knowledge graph for the above data under the impacts of the influence of components, industry applications costs and durability. Make sure that there are cross relationships and influences among the topics are well reflected.")
graph.visualize()
graph.description