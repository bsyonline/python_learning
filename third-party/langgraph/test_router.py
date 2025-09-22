from typing import Dict
from langgraph.graph import StateGraph, END, START

def classifier_node(state: Dict):
    """Classify the user input and determine next action."""
    messages = state.get("messages", [])
    next_action = ""
    last_message = messages[-1]['content'] if messages else ""
    
    if "weather" in last_message.lower():
        next_action = "weather"
    elif "calculator" in last_message.lower() or any(op in last_message for op in ['+', '-', '*', '/']):
        next_action = "calculator"
    else:
        next_action = "general"
    
    # Return updated state
    return {"messages": messages, "next_action": next_action}

def weather_node(state: Dict):
    """Simulate a weather tool."""
    messages = state.get("messages", [])
    response = "The weather today is sunny with a high of 25°C."
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}

def calculator_node(state: Dict):
    """Simulate a calculator tool."""
    messages = state.get("messages", [])
    last_message = messages[-1]['content'] if messages else ""
    # Simple calculation logic for demo purposes
    try:
        # This is a simplified example - in practice, you'd want to use a proper expression parser
        result = eval(last_message)  # Note: eval can be dangerous in production code
        response = f"The result of the calculation is: {result}"
    except:
        response = "I couldn't perform the calculation. Please check your input."
    
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}

def general_node(state: Dict):
    """Handle general queries."""
    messages = state.get("messages", [])
    last_message = messages[-1]['content'] if messages else ""
    response = f"I understand you're asking about '{last_message}'. How else can I help?"
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}

def router(state: Dict):
    """Route to appropriate node based on classification."""
    next_action = state.get("next_action", "general")
    return next_action

def create_graph():
    """Create and return the LangGraph workflow."""
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("calculator", calculator_node)
    workflow.add_node("general", general_node)
    
    # Set entry point
    workflow.add_edge(START, "classifier")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classifier",
        router,
        {
            "weather": "weather",
            "calculator": "calculator",
            "general": "general",
        }
    )
    
    # Add edges to end
    workflow.add_edge("weather", END)
    workflow.add_edge("calculator", END)
    workflow.add_edge("general", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def main():
    """Main function to run the demo."""
    app = create_graph()
    
    # Initialize state
    state = {"messages": [], "next_action": ""}
    
    # Add user messages for testing
    test_inputs = [
        "What's the weather like today?",
        "Calculate 15 * 3 + 7",
        "Tell me about artificial intelligence"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        state["messages"] = [{"role": "user", "content": user_input}]
        
        # Run the graph
        final_state = app.invoke(state)
        
        # Print the response
        for message in final_state["messages"]:
            if message["role"] == "assistant":
                print(f"Assistant: {message['content']}")


if __name__ == "__main__":
    main()