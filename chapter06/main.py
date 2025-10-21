from langgraph.types import Command


for chunk in graph.stream(Command(resume="Munich"), config):
    print(chunk)
